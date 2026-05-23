//! Synthetic-data accuracy test for the MF router.
//!
//! Generates 1000 examples where the "best stage" is a deterministic
//! function of the embedding (first two dims define quadrants → stage). The
//! MF router should learn this assignment with >80% accuracy.

use eoc_core::Stage;
use eoc_route_learned::matrix_factorization::MfRouter;
use eoc_route_learned::training::{Example, TrainingConfig, train_mf};
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

const DIM: usize = 8;

fn quadrant_stage(emb: &[f32]) -> Stage {
    let (a, b) = (emb[0], emb[1]);
    match (a >= 0.0, b >= 0.0) {
        (true, true) => Stage::Cache,
        (true, false) => Stage::Kv,
        (false, true) => Stage::Graph,
        (false, false) => Stage::Neural,
    }
}

fn synthesise(n: usize, seed: u64) -> Vec<Example> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let dist = Normal::new(0.0_f32, 1.0).expect("valid normal");
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let mut emb = vec![0.0_f32; DIM];
        for v in emb.iter_mut() {
            *v = dist.sample(&mut rng);
        }
        let stage = quadrant_stage(&emb);
        // Pure labels — the synthetic test focuses on classification accuracy,
        // not noise-handling.
        let success = true;
        // cost varies per stage so cost estimates are meaningful.
        let cost = match stage {
            Stage::Cache => 1_000,
            Stage::Kv => 50_000,
            Stage::Graph => 500_000,
            Stage::Neural => 50_000_000,
        };
        out.push(Example::new(emb, stage, success, cost));
    }
    out
}

#[test]
fn mf_router_accuracy_above_80() {
    let train = synthesise(1000, 17);
    let test = synthesise(200, 42);

    let cfg = TrainingConfig {
        epochs: 30,
        learning_rate: 0.1,
        l2: 1e-5,
        ..TrainingConfig::default()
    };
    let router: MfRouter = train_mf(&train, cfg);

    let mut correct = 0usize;
    for ex in &test {
        let (stage, _conf, _probs) = router.predict(&ex.embedding).expect("predict");
        if stage == ex.stage {
            correct += 1;
        }
    }
    let acc = correct as f32 / test.len() as f32;
    println!("MF synthetic accuracy: {:.3}", acc);
    assert!(acc > 0.8, "MF accuracy {acc} below 0.8 threshold");
}
