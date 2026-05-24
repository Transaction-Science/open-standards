//! Persist and load Ed25519 keypairs so a node has a stable PeerId
//! across restarts.
//!
//! On-disk format is the raw 32-byte Ed25519 secret seed. This keeps
//! the file format trivially diffable and avoids depending on PKCS#8 or
//! libp2p-protobuf serializations that have churned in the past.

use std::path::Path;

use libp2p::identity::{Keypair, ed25519};
use tokio::fs;
use tokio::io::AsyncWriteExt;

use crate::error::{Error, Result};

/// Generate a fresh Ed25519 keypair.
pub fn generate() -> Keypair {
    Keypair::generate_ed25519()
}

/// Save an Ed25519 keypair to `path`. The file is written with mode
/// 0600 on Unix systems.
pub async fn save_keypair(path: &Path, key: &Keypair) -> Result<()> {
    let ed = key
        .clone()
        .try_into_ed25519()
        .map_err(|e| Error::Identity(format!("not an ed25519 key: {e}")))?;
    let secret = ed.secret();
    let bytes: [u8; 32] = secret.as_ref().try_into().map_err(|_| {
        Error::Identity("ed25519 secret was not 32 bytes (libp2p invariant violated)".to_string())
    })?;

    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent).await?;
    }

    let mut opts = fs::OpenOptions::new();
    opts.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        opts.mode(0o600);
    }
    let mut f = opts.open(path).await?;
    f.write_all(&bytes).await?;
    f.flush().await?;
    Ok(())
}

/// Load an Ed25519 keypair previously written by [`save_keypair`].
pub async fn load_keypair(path: &Path) -> Result<Keypair> {
    let bytes = fs::read(path).await?;
    if bytes.len() != 32 {
        return Err(Error::Identity(format!(
            "expected 32-byte ed25519 secret, found {} bytes",
            bytes.len()
        )));
    }
    let mut seed = [0u8; 32];
    seed.copy_from_slice(&bytes);
    let secret = ed25519::SecretKey::try_from_bytes(&mut seed)
        .map_err(|e| Error::Identity(e.to_string()))?;
    let kp: ed25519::Keypair = secret.into();
    Ok(Keypair::from(kp))
}

/// Load the keypair at `path`, or generate + save a new one if the file
/// does not exist.
pub async fn load_or_create_keypair(path: &Path) -> Result<Keypair> {
    match fs::metadata(path).await {
        Ok(_) => load_keypair(path).await,
        Err(ref e) if e.kind() == std::io::ErrorKind::NotFound => {
            let kp = generate();
            save_keypair(path, &kp).await?;
            Ok(kp)
        }
        Err(e) => Err(Error::Io(e)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use libp2p::PeerId;
    use tempfile::TempDir;

    #[tokio::test]
    async fn roundtrip_preserves_peer_id() {
        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().join("key.bin");
        let kp = generate();
        let original_peer = PeerId::from(kp.public());
        save_keypair(&path, &kp).await.expect("save");
        let loaded = load_keypair(&path).await.expect("load");
        assert_eq!(PeerId::from(loaded.public()), original_peer);
    }

    #[tokio::test]
    async fn load_or_create_is_idempotent() {
        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().join("k").join("kp.bin");
        let first = load_or_create_keypair(&path).await.expect("create");
        let second = load_or_create_keypair(&path).await.expect("load");
        assert_eq!(PeerId::from(first.public()), PeerId::from(second.public()));
    }
}
