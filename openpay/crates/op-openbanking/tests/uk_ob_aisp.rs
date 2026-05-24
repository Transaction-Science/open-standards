//! UK Open Banking AISP integration test.
//!
//! Exercises the binding-layer URL builder, scope membership, and the
//! end-to-end shape of the AISP service trait via an in-memory mock.

use op_core::{Currency, Money};
use op_openbanking::aisp::{
    Account, AccountInfoService, AccountType, Balance, BalanceType, ConsentId, Transaction,
    TransactionCredit, TransactionStatus,
};
use op_openbanking::fapi::OAuth2Token;
use op_openbanking::uk_ob::{UkOpenBankingService, UkRwVersion, UkScope};
use op_openbanking::{Error, Result};

struct MockUkAisp {
    svc: UkOpenBankingService,
}

impl AccountInfoService for MockUkAisp {
    fn accounts(&self, _consent: &ConsentId, _token: &OAuth2Token) -> Result<Vec<Account>> {
        if !self.svc.has(UkScope::ReadAccountsDetail) {
            return Err(Error::ScopeNotAuthorised("ReadAccountsDetail".into()));
        }
        Ok(vec![Account {
            id: "acc-1".into(),
            identifier: "GB29NWBK60161331926819".into(),
            nickname: Some("Personal Current".into()),
            account_type: AccountType::Personal,
            subtype: None,
            currency: Currency::GBP,
        }])
    }

    fn balances(
        &self,
        _consent: &ConsentId,
        _token: &OAuth2Token,
        _account_id: &str,
    ) -> Result<Vec<Balance>> {
        if !self.svc.has(UkScope::ReadBalances) {
            return Err(Error::ScopeNotAuthorised("ReadBalances".into()));
        }
        Ok(vec![Balance {
            balance_type: BalanceType::InterimAvailable,
            amount: Money::from_minor(125_000, Currency::GBP),
            credit_debit: TransactionCredit::Credit,
            as_of: time::OffsetDateTime::UNIX_EPOCH,
        }])
    }

    fn transactions(
        &self,
        _consent: &ConsentId,
        _token: &OAuth2Token,
        _account_id: &str,
        _from: Option<time::OffsetDateTime>,
        _to: Option<time::OffsetDateTime>,
    ) -> Result<Vec<Transaction>> {
        if !self.svc.has(UkScope::ReadTransactionsDetail) {
            return Err(Error::ScopeNotAuthorised("ReadTransactionsDetail".into()));
        }
        Ok(vec![Transaction {
            id: "tx-1".into(),
            status: TransactionStatus::Booked,
            amount: Money::from_minor(2_500, Currency::GBP),
            credit_debit: TransactionCredit::Debit,
            booking_date: time::OffsetDateTime::UNIX_EPOCH,
            description: "Shop XYZ".into(),
            remittance: None,
        }])
    }
}

fn token() -> OAuth2Token {
    OAuth2Token {
        access_token: "tok".into(),
        token_type: "Bearer".into(),
        scopes: vec!["accounts".into()],
        expires_in: 3600,
        refresh_token: None,
        cert_thumbprint: None,
    }
}

#[test]
fn endpoint_root_is_v3_1() {
    let svc = UkOpenBankingService {
        version: UkRwVersion::V3_1_11,
        scopes: vec![UkScope::ReadAccountsDetail],
        aspsp_base_url: "https://api.aspsp.example".into(),
    };
    let url = svc.endpoint("/aisp/accounts");
    assert!(url.starts_with("https://api.aspsp.example/open-banking/v3.1/"));
    assert!(url.ends_with("/aisp/accounts"));
}

#[test]
fn aisp_happy_path_returns_account_balance_tx() {
    let svc = UkOpenBankingService {
        version: UkRwVersion::V3_1_11,
        scopes: vec![
            UkScope::ReadAccountsDetail,
            UkScope::ReadBalances,
            UkScope::ReadTransactionsDetail,
        ],
        aspsp_base_url: "https://api.aspsp.example".into(),
    };
    let mock = MockUkAisp { svc };
    let consent = ConsentId("consent-1".into());
    let tok = token();

    let accounts = mock.accounts(&consent, &tok).expect("accounts");
    assert_eq!(accounts.len(), 1);
    assert_eq!(accounts[0].currency, Currency::GBP);

    let balances = mock
        .balances(&consent, &tok, &accounts[0].id)
        .expect("balances");
    assert_eq!(balances.len(), 1);
    assert_eq!(balances[0].balance_type, BalanceType::InterimAvailable);

    let txs = mock
        .transactions(&consent, &tok, &accounts[0].id, None, None)
        .expect("transactions");
    assert_eq!(txs.len(), 1);
    assert_eq!(txs[0].credit_debit, TransactionCredit::Debit);
}

#[test]
fn missing_scope_fires_403() {
    let svc = UkOpenBankingService {
        version: UkRwVersion::V3_1_11,
        scopes: vec![UkScope::ReadAccountsDetail],
        aspsp_base_url: "x".into(),
    };
    let mock = MockUkAisp { svc };
    let err = mock
        .balances(&ConsentId("c".into()), &token(), "acc-1")
        .expect_err("scope guard");
    assert!(matches!(err, Error::ScopeNotAuthorised(_)));
}
