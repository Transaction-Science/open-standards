//! State import/export round-trip across all router families.

use eoc_route_learned::bandit::{LinUcbRouter, ThompsonSamplingRouter};
use eoc_route_learned::classifier::LogRegRouter;
use eoc_route_learned::matrix_factorization::MfRouter;
use eoc_route_learned::router::LearnedRouter;

#[test]
fn mf_roundtrip() {
    let router = MfRouter::new(8, 4, 11);
    let state = router.export_state();
    let restored = MfRouter::import_state(state).expect("import");
    assert_eq!(router.latent_dim(), restored.latent_dim());
    assert_eq!(router.embedding_dim(), restored.embedding_dim());
}

#[test]
fn logreg_roundtrip() {
    let router = LogRegRouter::new(8, 1e-4, 11);
    let state = router.export_state();
    let restored = LogRegRouter::import_state(state).expect("import");
    assert_eq!(router.embedding_dim(), restored.embedding_dim());
}

#[test]
fn linucb_roundtrip() {
    let router = LinUcbRouter::new(8, 1.0, 1.0);
    let state = router.export_state();
    let restored = LinUcbRouter::import_state(state).expect("import");
    assert_eq!(router.dim, restored.dim);
    assert_eq!(router.alpha, restored.alpha);
}

#[test]
fn thompson_roundtrip() {
    let router = ThompsonSamplingRouter::new(7);
    let state = router.export_state();
    let restored = ThompsonSamplingRouter::import_state(state).expect("import");
    assert_eq!(router.seed, restored.seed);
}

#[test]
fn rejects_wrong_algorithm_tag() {
    let r = MfRouter::new(4, 4, 0);
    let mut s = r.export_state();
    s.algorithm = "wrong".into();
    assert!(MfRouter::import_state(s).is_err());
}
