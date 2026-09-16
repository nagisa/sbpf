use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use solana_sbpf::{
    elf::Executable,
    program::{BuiltinProgram, SBPFVersion},
    verifier::RequisiteVerifier,
};
use test_utils::TestContextObject;
use std::{
    fs,
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
    convert::TryInto,
};

type DynError = Box<dyn std::error::Error + Send + Sync + 'static>;

#[derive(Debug, Clone)]
struct Sample {
    path: PathBuf,
    size_bytes: usize,
}

fn collect_sos(dir: &Path) -> Result<Vec<Sample>, DynError> {
    let mut out = Vec::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("so") {
            continue;
        }
        let data = fs::read(&path)?;
        out.push(Sample {
            path,
            size_bytes: data.len(),
        });
    }
    Ok(out)
}

fn run_standard_jit(executable: &Executable<TestContextObject>, rounds: usize) -> (bool, u128, usize) {
    let mut total_us = 0u128;
    let mut machine_code_len = 0usize;
    let mut mem_size = 0usize;

    for round in 0..rounds {
        let start = Instant::now();
        if executable.jit_compile().is_err() {
            return (false, total_us, machine_code_len);
        }
        if round == 0 {
            continue;
        }
        total_us += start.elapsed().as_micros();

        if let Some(compiled) = executable.get_compiled_program() {
            machine_code_len = compiled.machine_code_length();
        }
        let _ = executable.take_compiled_program();
    }

    (true, total_us / (rounds as u128 - 1), machine_code_len)
}

fn run_token_threading_jit(executable: &Executable<TestContextObject>, rounds: usize) -> (bool, u128, usize) {
    let mut total_us = 0u128;
    let mut pc_len = 0usize;
    let mut code_len = 0usize;

    for round in 0..rounds {
        let start = Instant::now();
        let code = match solana_sbpf::token_threading::compile(&executable) {
            Ok((_, code)) => {
                code
            }
            Err(e) => {
                eprintln!("tokenjit error: {e:?}");
                return (false, 0, 0);
            }
        };
        if round == 0 {
            continue;
        }
        total_us += start.elapsed().as_micros();
        code_len = code.len();
    }
    (true, total_us / (rounds as u128 - 1), code_len)
}

fn main() -> Result<(), DynError> {
    let mut args = std::env::args().skip(1);
    let dir = PathBuf::from(args.next().ok_or("usage: jit_compare <dir> [rounds]")?);
    let rounds: usize = args.next().as_deref().unwrap_or("5").parse()?;

    let mut samples = collect_sos(&dir)?;
    let mut rng = StdRng::seed_from_u64(42);
    samples.shuffle(&mut rng);

    println!("path,size_bytes,ok,ver,standard_compile_us,standard_machine_code_len,token_compile_us,token_machine_code_len");

    for sample in samples {
        let bytes = fs::read(&sample.path)?;
        let loader = Arc::new(BuiltinProgram::new_mock());
        let executable = match Executable::<TestContextObject>::from_elf(&bytes, loader) {
            Ok(x) => x,
            Err(_) => continue,
        };
        if executable.verify::<RequisiteVerifier>().is_err() || executable.get_sbpf_version() != SBPFVersion::V3 {
            continue
        }

        let (token_ok, token_compile_us, token_code_len) =
            run_token_threading_jit(&executable, rounds);
        let (standard_ok, standard_compile_us, standard_machine_code_len) =
            run_standard_jit(&executable, rounds);
        let path = sample.path.file_stem().unwrap().to_string_lossy();

        println!(
            "{},{},{},{:?},{},{},{},{}",
            path,
            sample.size_bytes,
            standard_ok && token_ok,
            executable.get_sbpf_version(),
            standard_compile_us,
            standard_machine_code_len,
            token_compile_us,
            token_code_len
        );
    }

    Ok(())
}

fn csv_escape(s: &str) -> String {
    let escaped = s.replace('"', "\"\"");
    format!("\"{escaped}\"")
}
