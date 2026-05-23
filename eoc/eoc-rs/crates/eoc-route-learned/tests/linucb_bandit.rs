//! 100-round LinUCB synthetic bandit test.
//!
//! Four arms with known mean rewards. Context is a fixed-length embedding
//! that *informs* but doesn't determine reward (Gaussian noise). After ~50
//! rounds the router should pull the best arm more than any other.

use eoc_core::Stage;
use eoc_route_learned::bandit::LinUcbRouter;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use std::collections::HashMap;

const DIM: usize = 4;

fn arm_mean(stage: Stage) -> f32 {
    match stage {
        Stage::Cache => 0.10,
        Stage::Kv => 0.20,
        Stage::Graph => 0.30,
        // Best arm: Neural.
        Stage::Neural => 0.80,
    }
}

#[test]
fn linucb_converges_to_best_arm() {
    // Lower alpha → less exploration, more exploitation. Ridge=0.1 so the
    // estimated theta moves quickly.
    let mut router = LinUcbRouter::new(DIM, 0.1, 0.1);
    let mut rng = rand::rngs::StdRng::seed_from_u64(0xBEEF);
    let ctx_dist = Normal::new(0.0_f32, 1.0).expect("valid normal");
    let reward_noise = Normal::new(0.0_f32, 0.02).expect("valid normal");

    let mut pulls: HashMap<Stage, usize> = HashMap::new();
    let mut tail_pulls: HashMap<Stage, usize> = HashMap::new();

    for round in 0..100 {
        let ctx: Vec<f32> = (0..DIM).map(|_| ctx_dist.sample(&mut rng)).collect();
        let (pick, _score) = router.pick(&ctx).expect("pick");
        *pulls.entry(pick).or_insert(0) += 1;
        if round >= 50 {
            *tail_pulls.entry(pick).or_insert(0) += 1;
        }
        let reward = (arm_mean(pick) + reward_noise.sample(&mut rng)).clamp(0.0, 1.0);
        router.update(&ctx, pick, reward).expect("update");
    }

    println!("LinUCB pulls (all 100 rounds): {pulls:?}");
    println!("LinUCB pulls (rounds 50-99): {tail_pulls:?}");

    let neural_tail = *tail_pulls.get(&Stage::Neural).unwrap_or(&0);
    let best_other = [Stage::Cache, Stage::Kv, Stage::Graph]
        .iter()
        .map(|s| *tail_pulls.get(s).unwrap_or(&0))
        .max()
        .unwrap_or(0);
    assert!(
        neural_tail >= best_other,
        "neural ({neural_tail}) should be pulled at least as often as any other ({best_other}) after round 50"
    );
    // And neural should dominate over the full 100 rounds, since its mean
    // reward is 0.8 vs ≤0.3 for every other arm.
    let neural_total = *pulls.get(&Stage::Neural).unwrap_or(&0);
    assert!(
        neural_total >= 40,
        "neural should be pulled ≥40 times over 100 rounds, got {neural_total}"
    );
}
