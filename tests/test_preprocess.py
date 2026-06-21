import pandas as pd

from src.preprocess import clean_text, filter_and_clean, normalize_product


def test_normalize_product():
    assert normalize_product("Credit card or prepaid card") == "Credit Card"
    assert normalize_product("Checking or savings account") == "Savings Account"
    assert normalize_product("Money transfer, virtual currency, or money service") == "Money Transfer"


def test_clean_text_removes_boilerplate_and_symbols():
    text = "I am writing to file a complaint!!! XXXX about billing @@@"
    cleaned = clean_text(text)
    assert "i am writing to file a complaint" not in cleaned
    assert "xxxx" not in cleaned
    assert "@@@" not in cleaned


def test_filter_and_clean():
    df = pd.DataFrame({
        "Complaint ID": [1, 2, 3],
        "Product": ["Credit card", "Mortgage", "Personal loan"],
        "Consumer complaint narrative": ["Billing problem happened repeatedly", "House issue", "Loan payment issue happened"],
    })
    out = filter_and_clean(df)
    assert len(out) == 2
    assert set(out["product_category"]) == {"Credit Card", "Personal Loan"}
    assert "cleaned_narrative" in out.columns
