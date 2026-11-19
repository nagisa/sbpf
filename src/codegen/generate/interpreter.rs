//! Generation of the interpreter.

use super::{Generator, LabelRelocs, PatchFields, Reg, TemplateOpcode};
use crate::codegen::arch::{self, SupportEntry, SupportingCode, TemplateRelocationKind};
use crate::ebpf;
use crate::memory_management::{Mapping, ReadExecute, HUGE_PAGE_SIZE};
use crate::program::SBPFVersion;
use dynasmrt::relocations::{Relocation, RelocationKind};
use dynasmrt::DynamicLabel;
#[cfg(feature = "codegen-debug")]
use std::convert::TryInto;
use std::sync::LazyLock;

/// The interpreter step for the instructions with `opcode`, of the interpreter for the SBPF
/// `version`.
pub(in crate::codegen) fn interpreter_step(
    version: SBPFVersion,
    opcode: TemplateOpcode,
) -> *const u8 {
    &interpreter(version).steps[InterpreterGenerator::step_offset(opcode)]
}

pub(in crate::codegen) struct Interpreter {
    /// The steps, followed by the copy of the supporting code they call.
    pub(in crate::codegen) steps: Mapping<ReadExecute>,
    /// The length of the code of each step, which is followed by padding to the size of the step.
    #[cfg(feature = "codegen-debug")]
    pub(in crate::codegen) step_lens: Box<[u8]>,
}

impl Interpreter {
    /// The copy of the supporting code.
    pub(in crate::codegen) fn supports(&self) -> *const u8 {
        self.steps[InterpreterGenerator::SUPPORTS..].as_ptr()
    }
}

/// Generates the interpreter.
///
/// This interpreter uses a dispatch table with one step of `1 << STEP_SIZE_LOG2` bytes per
/// `TemplateOpcode`, each running its instruction and then dispatching the execution to the next
/// one directly.
///
/// A step that does not fit its slot together with the dispatch jumps to a copy of the dispatch in
/// a slot of its own, [`Self::dispatch_slot`], one that no valid instruction uses. Each such step
/// has its own copy, so that the indirect jumps of the dispatch remain as many as the steps for the
/// branch predictor.
pub(in crate::codegen) struct InterpreterGenerator {
    buffer: Mapping,
    relocs: LabelRelocs<arch::Relocation>,
    /// The offsets of the entries of the latest island.
    island: Option<arch::IslandEntries>,
    /// The offsets of the entries of the supporting code.
    supports: arch::SupportEntries,
    offset: usize,
    version: SBPFVersion,
    opcode: TemplateOpcode,
    /// Is the code generated for this instruction terminal?
    ///
    /// No further instructions other than the epilogue expected to appear after this point.
    terminal: bool,
}

impl InterpreterGenerator {
    pub(in crate::codegen) const STEP_SIZE_LOG2: u8 = arch::INTERPRETER_STEP_SIZE_LOG2;
    pub(in crate::codegen) const STEP_TABLE_SIZE: usize = 0x1_0000 * (1 << Self::STEP_SIZE_LOG2);
    /// The offset of the supporting code, right after the steps.
    const SUPPORTS: usize = Self::STEP_TABLE_SIZE;

    /// Offset into the `buffer` for this opcode.
    pub(in crate::codegen) fn step_offset(opcode: TemplateOpcode) -> usize {
        opcode.index() << Self::STEP_SIZE_LOG2
    }

    /// The slot of the dispatch of the step for `opcode`, for when the step does not fit its own
    /// slot together with the dispatch.
    ///
    /// The slot has a register field that no valid instruction has: these are more than those of
    /// the valid instructions, which are mapped to them in order. Its operation differs in the
    /// lowest bit, which puts it into the cache set next to that of the step, rather than the same
    /// one, as long as the slots are laid out in the order of the opcodes. Operation 0 is not a valid
    /// instruction, and its slots hold the islands instead, so operation 1 (not a valid instruction
    /// either) keeps its own.
    fn dispatch_slot(opcode: TemplateOpcode) -> Option<TemplateOpcode> {
        let (Some(dst), Some(src)) = (opcode.dst(), opcode.src()) else {
            return None;
        };
        let op = match opcode.op() {
            0 => return None,
            1 => 1,
            op => op ^ 1,
        };
        let registers = usize::from(dst.0)
            .checked_mul(Reg::COUNT)?
            .checked_add(usize::from(src.0))?;
        let field = u16::from(INVALID_REGISTER_FIELDS[registers]);
        Some(TemplateOpcode(u16::from(op) | field << 8))
    }

    /// The step whose dispatch is in the slot of `opcode`, see [`Self::dispatch_slot`].
    pub(in crate::codegen) fn dispatched_step(opcode: TemplateOpcode) -> Option<TemplateOpcode> {
        let op = match opcode.op() {
            0 => return None,
            1 => 1,
            op => op ^ 1,
        };
        let field = (opcode.0 >> 8) as u8;
        let registers = INVALID_REGISTER_FIELDS[..Reg::COUNT * Reg::COUNT]
            .iter()
            .position(|&invalid| invalid == field)?;
        let (dst, src) = (registers / Reg::COUNT, registers % Reg::COUNT);
        Some(TemplateOpcode(
            u16::from(op) | (dst as u16) << 8 | (src as u16) << 12,
        ))
    }

    pub(in crate::codegen) fn base(&self) -> *const u8 {
        self.buffer.as_ptr()
    }

    fn new(version: SBPFVersion, mut buffer: Mapping) -> Self {
        let supports = SupportingCode::get();
        let supports_end = Self::SUPPORTS.checked_add(supports.code.len()).unwrap();
        assert!(buffer.len() >= supports_end);
        const { assert!(Self::SUPPORTS.is_multiple_of(arch::SUPPORTS_ALIGNMENT)) };
        buffer[Self::SUPPORTS..supports_end].copy_from_slice(&supports.code);
        Self {
            version,
            buffer,
            relocs: LabelRelocs::new(),
            island: None,
            supports: supports.at(Self::SUPPORTS),
            offset: 0,
            opcode: TemplateOpcode(0),
            terminal: false,
        }
    }
}

impl InterpreterGenerator {
    /// Complete the code of the `slot` that ends at the current offset, returning its length.
    fn finish_slot(&mut self, slot: TemplateOpcode) -> usize {
        let start = Self::step_offset(slot);
        let len = self.offset.checked_sub(start).unwrap();
        assert!(
            len <= 1 << Self::STEP_SIZE_LOG2,
            "step for {:#x} is too long",
            slot.0
        );
        let code = &mut self.buffer[start..self.offset];
        self.relocs.resolve(code, start);
        len
    }

    /// Resolve the relocation that `dynasm` reports at the current offset for an `extern` jump
    /// target, which in the interpreter is the offset of the `target` rather than its address.
    pub(in crate::codegen) fn value_reloc(
        &mut self,
        target: usize,
        field_offset: u8,
        ref_offset: u8,
        kind: u8,
    ) {
        // `dynasm` reports a relocation relative to an absolute address, while the offset is
        // reached like a label: clear the kind to `RelocationKind::Relative`.
        self.relative_reloc(target, 0, field_offset, ref_offset, kind & 0x3f);
    }

    /// Resolve a relative relocation reported at the current offset to the `target` offset.
    pub(in crate::codegen) fn relative_reloc(
        &mut self,
        target: usize,
        target_offset: isize,
        field_offset: u8,
        ref_offset: u8,
        kind: u8,
    ) {
        let patch =
            PatchFields::<arch::Relocation>::new(target_offset, field_offset, ref_offset, kind)
                .at(self.offset);
        assert!(
            matches!(patch.relocation.kind(), RelocationKind::Relative),
            "unsupported relocation"
        );
        let base = self.buffer.as_ptr() as usize;
        let field = &mut self.buffer[patch.range(0)];
        patch
            .patch(field, base, target)
            .expect("relocation target out of reach of the interpreter");
    }
}

impl Generator for InterpreterGenerator {
    fn extend(&mut self, buffer: &[u8]) {
        assert!(!self.terminal);
        let step_capacity = 1usize << Self::STEP_SIZE_LOG2;
        let remaining_capacity = step_capacity
            .checked_sub(self.offset % step_capacity)
            .unwrap();
        assert!(buffer.len() <= remaining_capacity, "step is too long!");
        let end = self.offset.checked_add(buffer.len()).unwrap();
        self.buffer[self.offset..end].copy_from_slice(buffer);
        self.offset = end;
    }

    fn offset(&self) -> usize {
        self.offset
    }

    fn global_reloc(
        &mut self,
        name: &'static str,
        target_offset: isize,
        field_offset: u8,
        ref_offset: u8,
        kind: u8,
    ) {
        let entry = arch::IslandEntry::of_global(name)
            .unwrap_or_else(|| panic!("global reference to an unknown symbol {}", name));
        let island = self
            .island
            .expect("the island precedes the code that refers to it");
        let target = island[entry as usize];
        self.relative_reloc(target, target_offset, field_offset, ref_offset, kind);
    }

    fn support_reloc(
        &mut self,
        support: SupportEntry,
        target_offset: isize,
        field_offset: u8,
        ref_offset: u8,
        encoding: u8,
    ) {
        let target = self.supports[support as usize];
        self.relative_reloc(target, target_offset, field_offset, ref_offset, encoding);
    }

    fn template_reloc(&mut self, _: TemplateRelocationKind, _: isize, _: u8, _: u8, _: u8) {
        // Intentionally empty: interpreter does not generate templates.
    }

    fn dynamic_reloc(
        &mut self,
        id: DynamicLabel,
        target_offset: isize,
        field_offset: u8,
        ref_offset: u8,
        kind: u8,
    ) {
        let patch = PatchFields::new(target_offset, field_offset, ref_offset, kind);
        self.relocs.dynamic_reloc(self.offset, id, patch);
    }

    fn new_dynamic_label(&mut self) -> DynamicLabel {
        self.relocs.new_dynamic_label()
    }

    fn dynamic_label(&mut self, id: DynamicLabel) {
        self.relocs.dynamic_label(id, self.offset);
    }

    fn version(&self) -> SBPFVersion {
        self.version
    }

    fn opcode(&self) -> TemplateOpcode {
        self.opcode
    }

    fn bpf_taken_branch(&mut self, next_insn_in_temp: bool) {
        arch::interpreter_taken_branch(self, next_insn_in_temp);
        self.terminal = true;
    }

    fn meter_checked(&mut self) {
        // Intentionally empty: every step checks the meter.
    }
}

static INTERPRETERS: [LazyLock<Interpreter>; 5] = [
    LazyLock::new(|| generate_interpreter(SBPFVersion::V0)),
    LazyLock::new(|| generate_interpreter(SBPFVersion::V1)),
    LazyLock::new(|| generate_interpreter(SBPFVersion::V2)),
    LazyLock::new(|| generate_interpreter(SBPFVersion::V3)),
    LazyLock::new(|| generate_interpreter(SBPFVersion::V4)),
];

/// The register fields of the opcodes, `dst | src << 4`, that no valid instruction has, in order.
const INVALID_REGISTER_FIELDS: [u8; 256 - Reg::COUNT * Reg::COUNT] = {
    let mut fields = [0; 256 - Reg::COUNT * Reg::COUNT];
    let (mut field, mut count) = (0, 0);
    while field < 256 {
        if field & 0xf >= Reg::COUNT || field >> 4 >= Reg::COUNT {
            fields[count] = field as u8;
            count += 1;
        }
        field += 1;
    }
    fields
};

/// The interpreter for the SBPF `version`.
pub(in crate::codegen) fn interpreter(version: SBPFVersion) -> &'static Interpreter {
    &INTERPRETERS[version as usize]
}

fn generate_interpreter(version: SBPFVersion) -> Interpreter {
    let len = InterpreterGenerator::SUPPORTS
        .checked_add(SupportingCode::get().code.len())
        .unwrap();
    #[cfg(feature = "codegen-debug")]
    let mut debug_image =
        super::super::debug::DebugImage::new(&format!("interpreter-{version:?}").to_lowercase());
    // Whole huge pages, so that the supporting code after the steps is in one too.
    let len = len.next_multiple_of(HUGE_PAGE_SIZE);
    let mut buffer = cfg_select! {
        feature = "codegen-debug" => debug_image.map_sub2g(len),
        _ => Mapping::sub2g(len),
    };
    // The space after the code of a step traps if it is ever executed.
    buffer.fill(arch::TRAP_FILL);
    let mut generator = InterpreterGenerator::new(version, buffer);
    #[cfg(feature = "codegen-debug")]
    let mut step_lens = vec![0; TemplateOpcode::COUNT];
    // The islands, and the copies of the dispatch, which tell how long the dispatch is for the
    // instructions of each size.
    let mut islands = Vec::with_capacity(TemplateOpcode::COUNT >> 8);
    let mut dispatch_lens = [0; 3];
    for slot in TemplateOpcode::all() {
        generator.offset = InterpreterGenerator::step_offset(slot);
        generator.terminal = false;
        if slot.op() == 0 {
            // Not a valid instruction, so it only terminates the execution and needs no dispatch.
            generator.opcode = slot;
            let start = generator.offset;
            let island = arch::island(&mut generator).map(|entry| start.wrapping_add(entry));
            generator.island = Some(island);
            islands.push(island);
            arch::bpf_insn(&mut generator);
        } else if let Some(step) = InterpreterGenerator::dispatched_step(slot) {
            generator.opcode = step;
            let start = generator.offset;
            arch::interpreter_dispatch(&mut generator);
            let size = ebpf::opcode_size(step.op(), version) / ebpf::INSN_SIZE;
            dispatch_lens[size] = generator.offset.checked_sub(start).unwrap();
        } else {
            continue;
        }
        let len = generator.finish_slot(slot);
        #[cfg(feature = "codegen-debug")]
        {
            step_lens[slot.index()] = len.try_into().unwrap();
        }
        #[cfg(not(feature = "codegen-debug"))]
        let _ = len;
    }

    for opcode in TemplateOpcode::all() {
        if opcode.op() == 0 || InterpreterGenerator::dispatched_step(opcode).is_some() {
            continue;
        }
        let start = InterpreterGenerator::step_offset(opcode);
        generator.opcode = opcode;
        generator.offset = start;
        generator.terminal = false;
        generator.island = Some(islands[opcode.index() >> 8]);
        arch::bpf_insn(&mut generator);
        generator.terminal = false;
        let size = ebpf::opcode_size(opcode.op(), version) / ebpf::INSN_SIZE;
        let fits = generator
            .offset
            .checked_sub(start)
            .unwrap()
            .checked_add(dispatch_lens[size])
            .unwrap()
            <= 1 << InterpreterGenerator::STEP_SIZE_LOG2;
        if opcode.dst().is_none() || opcode.src().is_none() {
            // Not a valid instruction, so it only terminates the execution and needs no dispatch.
        } else if fits {
            arch::interpreter_dispatch(&mut generator);
        } else {
            let slot = InterpreterGenerator::dispatch_slot(opcode).unwrap();
            arch::interpreter_jump(&mut generator, InterpreterGenerator::step_offset(slot));
        }
        let len = generator.finish_slot(opcode);
        #[cfg(feature = "codegen-debug")]
        {
            step_lens[opcode.index()] = len.try_into().unwrap();
        }
        #[cfg(not(feature = "codegen-debug"))]
        let _ = len;
    }

    let interpreter = Interpreter {
        steps: generator.buffer.protect(),
        #[cfg(feature = "codegen-debug")]
        step_lens: step_lens.into_boxed_slice(),
    };
    // The steps of a program are spread over the whole table, too many 4 KiB pages for the TLB.
    interpreter.steps.collapse_huge_pages();
    #[cfg(feature = "codegen-debug")]
    super::super::debug::finish_interpreter(debug_image, version, &interpreter);
    interpreter
}
