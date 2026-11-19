use preflate_container::{
    PreflateContainerConfig, PreflateContainerProcessor, ProcessBuffer, RecreateContainerProcessor,
};
use std::{
    env,
    error::Error,
    io::{self, Cursor, Read, Write},
};

const MAX_OBJECT_SIZE: usize = 8 << 20;
const MAX_PLAINTEXT_CHUNK: usize = 32 << 20;

fn recreate(payload: &[u8], size: usize) -> Result<Vec<u8>, Box<dyn Error>> {
    let mut data = vec![0; size];
    let mut output = Cursor::new(data.as_mut_slice());
    RecreateContainerProcessor::new(MAX_PLAINTEXT_CHUNK).process_buffer(
        payload,
        true,
        &mut output,
    )?;
    if output.position() != size as u64 {
        return Err("Preflate produced an unexpected output size".into());
    }
    Ok(data)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    let mut input = Vec::new();
    io::stdin()
        .take((MAX_OBJECT_SIZE + 1) as u64)
        .read_to_end(&mut input)?;
    if input.is_empty() || input.len() > MAX_OBJECT_SIZE {
        return Err("Preflate input is outside ARIL's object size limit".into());
    }
    let result = match args.as_slice() {
        [mode] if mode == "encode" => {
            let config = PreflateContainerConfig {
                chunk_plain_text_limit: MAX_PLAINTEXT_CHUNK,
                ..PreflateContainerConfig::default()
            };
            let mut payload = Vec::new();
            PreflateContainerProcessor::new(&config, 9, false).process_buffer(
                &input,
                true,
                &mut payload,
            )?;
            if payload.len() >= input.len() || recreate(&payload, input.len())? != input {
                return Err("Preflate did not yield a smaller, byte-exact result".into());
            }
            payload
        }
        [mode, size] if mode == "decode" => {
            let size = size.parse::<usize>()?;
            if size == 0 || size > MAX_OBJECT_SIZE {
                return Err("Invalid ARIL Preflate output size".into());
            }
            recreate(&input, size)?
        }
        _ => return Err("Usage: aril-preflate encode | decode SIZE".into()),
    };
    io::stdout().write_all(&result)?;
    Ok(())
}
