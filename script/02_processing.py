import os
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

# cau hinh duong dan
DATA_DIR   = "./data"
OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# buoc 1: doc du lieu chinh
def load_main_table(path: str) -> pd.DataFrame:
    """Doc bang chinh application_train.csv."""
    df = pd.read_csv(path)
    print(f"[INFO] Shape ban dau: {df.shape}")
    print(f"[INFO] Ti le TARGET=1: {df['TARGET'].mean():.4f} ({df['TARGET'].sum()} mau)")
    return df


# buoc 2: xu ly gia tri bat thuong
def handle_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Phat hien va xu ly cac gia tri bat thuong.
    DAYS_EMPLOYED = 365243 la ma loi cua he thong, chuyen ve NaN.
    """
    anomaly_mask = df["DAYS_EMPLOYED"] == 365243
    print(f"[INFO] DAYS_EMPLOYED = 365243: {anomaly_mask.sum()} dong -> chuyen thanh NaN")
    df.loc[anomaly_mask, "DAYS_EMPLOYED"] = np.nan
    df["FLAG_DAYS_EMPLOYED_ANOMALY"] = anomaly_mask.astype(int)
    return df


# buoc 3: tao dac trung moi (feature engineering)
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tao cac dac trung moi co y nghia kinh doanh cao."""
    # cac bien co ban
    df["YEARS_BIRTH"]            = (-df["DAYS_BIRTH"] / 365).round(1)
    df["ANNUITY_INCOME_RATIO"]   = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] + 1e-9)
    df["CREDIT_INCOME_RATIO"]    = df["AMT_CREDIT"] / (df["AMT_INCOME_TOTAL"] + 1e-9)
    df["CREDIT_GOODS_RATIO"]     = df["AMT_CREDIT"] / (df["AMT_GOODS_PRICE"] + 1e-9)
    df["INCOME_PER_PERSON"]      = df["AMT_INCOME_TOTAL"] / (df["CNT_FAM_MEMBERS"] + 1e-9)
    df["EMPLOYED_TO_BIRTH_RATIO"]= df["DAYS_EMPLOYED"] / (df["DAYS_BIRTH"] + 1e-9)

    # bien bo sung
    # ty le dong tien tra no hang thang tren tong khoan vay
    df["CREDIT_TERM"]            = df["AMT_ANNUITY"] / (df["AMT_CREDIT"] + 1e-9)
    # ty le tham nien lam viec tren tuoi doi
    df["WORKING_LIFE_RATIO"]     = df["DAYS_EMPLOYED"] / (df["DAYS_BIRTH"] + 1e-9)
    # chi so tin dung tong hop (tuong tac phi tuyen tinh giua 3 nguon EXT)
    df["EXT_SOURCES_PROD"]       = df["EXT_SOURCE_1"] * df["EXT_SOURCE_2"] * df["EXT_SOURCE_3"]
    df["EXT_SOURCES_MEAN"]       = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(axis=1)
    # kha nang chi tra thuc te
    df["INCOME_TO_ANNUITY_RATIO"]= df["AMT_INCOME_TOTAL"] / (df["AMT_ANNUITY"] + 1e-9)

    print(f"[INFO] Hoan tat feature engineering: {len(df.columns)} bien.")
    return df


# buoc 4: tich hop bang phu (merging)
def aggregate_bureau(bureau_path: str) -> pd.DataFrame:
    """
    Tong hop thong tin tu bureau.csv:
    - Tong so khoan vay tin dung cu
    - So no qua han trung binh
    - So khoan vay dang Active
    """
    if not os.path.exists(bureau_path):
        print(f"[WARN] Khong tim thay {bureau_path} - bo qua bang bureau")
        return None

    bureau = pd.read_csv(bureau_path)
    agg = bureau.groupby("SK_ID_CURR").agg(
        BUREAU_LOAN_COUNT      = ("SK_ID_BUREAU", "count"),
        BUREAU_AVG_OVERDUE     = ("AMT_CREDIT_SUM_OVERDUE", "mean"),
        BUREAU_ACTIVE_COUNT    = ("CREDIT_ACTIVE", lambda x: (x == "Active").sum()),
        BUREAU_CLOSED_COUNT    = ("CREDIT_ACTIVE", lambda x: (x == "Closed").sum()),
        BUREAU_MAX_DAY_OVERDUE = ("CREDIT_DAY_OVERDUE", "max"),
    ).reset_index()

    print(f"[INFO] bureau agg shape: {agg.shape}")
    return agg


def aggregate_previous_application(prev_path: str) -> pd.DataFrame:
    """
    Tong hop thong tin tu previous_application.csv:
    - Ti le don vay tung bi tu choi
    - So don vay truoc day
    - AMT_APPLICATION trung binh
    """
    if not os.path.exists(prev_path):
        print(f"[WARN] Khong tim thay {prev_path} - bo qua bang previous_application")
        return None

    prev = pd.read_csv(prev_path)
    agg = prev.groupby("SK_ID_CURR").agg(
        PREV_APP_COUNT           = ("SK_ID_PREV", "count"),
        PREV_REFUSED_RATIO       = ("NAME_CONTRACT_STATUS", lambda x: (x == "Refused").sum() / len(x)),
        PREV_APPROVED_RATIO      = ("NAME_CONTRACT_STATUS", lambda x: (x == "Approved").sum() / len(x)),
        PREV_AVG_AMT_APPLICATION = ("AMT_APPLICATION", "mean"),
        PREV_AVG_AMT_CREDIT      = ("AMT_CREDIT", "mean"),
    ).reset_index()

    print(f"[INFO] previous_application agg shape: {agg.shape}")
    return agg


def merge_auxiliary_tables(df: pd.DataFrame) -> pd.DataFrame:
    """Gop cac bang phu vao bang chinh."""
    bureau_agg = aggregate_bureau(os.path.join(DATA_DIR, "bureau.csv"))
    if bureau_agg is not None:
        df = df.merge(bureau_agg, on="SK_ID_CURR", how="left")
        print(f"[INFO] Sau merge bureau: {df.shape}")

    prev_agg = aggregate_previous_application(os.path.join(DATA_DIR, "previous_application.csv"))
    if prev_agg is not None:
        df = df.merge(prev_agg, on="SK_ID_CURR", how="left")
        print(f"[INFO] Sau merge previous_application: {df.shape}")

    return df


# buoc 5: ma hoa bien phan loai (encoding)
def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ma hoa bien phan loai:
    - Binary (2 gia tri)  -> Label Encoding (0/1)
    - Multi-class         -> One-Hot Encoding (get_dummies)
    """
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    print(f"[INFO] So cot phan loai: {len(cat_cols)}")

    for col in cat_cols:
        if df[col].nunique() <= 2:
            df[col] = pd.factorize(df[col])[0]
        else:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True, dtype=int)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)

    print(f"[INFO] Shape sau encoding: {df.shape}")
    return df


# buoc 6: dien gia tri thieu & chuan hoa
def impute_and_scale(X_train: pd.DataFrame, X_test: pd.DataFrame = None):
    """
    - SimpleImputer: median cho so, most_frequent cho chuoi
    - StandardScaler: chuan hoa Z-score
    Luu scaler ra file scaler_master.pkl de dung lai o inference.
    """
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

    # imputer so
    num_imputer = SimpleImputer(strategy="median")
    X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
    if X_test is not None:
        X_test[num_cols] = num_imputer.transform(X_test[num_cols])

    # imputer phan loai
    if cat_cols:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        X_train[cat_cols] = cat_imputer.fit_transform(X_train[cat_cols])
        if X_test is not None:
            X_test[cat_cols] = cat_imputer.transform(X_test[cat_cols])

    # scaler
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    if X_test is not None:
        X_test[num_cols] = scaler.transform(X_test[num_cols])

    scaler_path = os.path.join(OUTPUT_DIR, "scaler_master.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"[INFO] Da luu scaler tai: {scaler_path}")

    return X_train, X_test


# buoc 7: can bang nhan (imbalanced learning)
def balance_classes(X: pd.DataFrame, y: pd.Series, strategy: str = "smote"):
    """
    Can bang lai ti le nhan TARGET (hien tai ~8% la default).

    strategy:
        'smote'       -> Tao mau gia (tong hop) cho lop thieu so
        'undersample' -> Loai bo ngau nhien mau lop da so
        'combined'    -> Ket hop SMOTE + RandomUnderSampler (khuyen dung)
    """
    print(f"[INFO] Can bang nhan voi chien luoc: {strategy}")
    print(f"[INFO] Truoc: TARGET=0: {(y==0).sum()}, TARGET=1: {(y==1).sum()}")

    if strategy == "smote":
        sampler = SMOTE(random_state=42, k_neighbors=5)
        X_res, y_res = sampler.fit_resample(X, y)

    elif strategy == "undersample":
        sampler = RandomUnderSampler(random_state=42, sampling_strategy=0.5)
        X_res, y_res = sampler.fit_resample(X, y)

    elif strategy == "combined":
        # SMOTE tang lop thieu so len 20%, sau do undersample lop da so xuong 1:2
        pipeline = ImbPipeline([
            ("smote",      SMOTE(random_state=42, sampling_strategy=0.2)),
            ("undersample", RandomUnderSampler(random_state=42, sampling_strategy=0.5)),
        ])
        X_res, y_res = pipeline.fit_resample(X, y)

    else:
        raise ValueError(f"strategy khong hop le: {strategy}")

    print(f"[INFO] Sau: TARGET=0: {(y_res==0).sum()}, TARGET=1: {(y_res==1).sum()}")
    return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name="TARGET")


# pipeline chinh
def run_preprocessing(
    train_path: str = None,
    balance_strategy: str = "combined",
    drop_leakage: bool = True,
    corr_threshold: float = 0.9,
):
    """
    Chay toan bo pipeline tien xu ly.

    Args:
        train_path       : Duong dan toi application_train.csv
        balance_strategy : Chien luoc can bang nhan ('smote'/'undersample'/'combined')
        drop_leakage     : Co loai bo cot tuong quan cao (>corr_threshold) khong
        corr_threshold   : Nguong tuong quan de loai bo

    Returns:
        X_res, y_res, feature_names
    """
    if train_path is None:
        train_path = os.path.join(DATA_DIR, "application_train.csv")

    df = load_main_table(train_path)
    df = handle_anomalies(df)
    df = engineer_features(df)
    df = merge_auxiliary_tables(df)
    df = encode_categoricals(df)

    drop_cols   = ["SK_ID_CURR", "TARGET"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].copy()
    y = df["TARGET"].copy()

    # kiem tra data leakage qua tuong quan
    if drop_leakage:
        print(f"[INFO] Kiem tra Data Leakage (nguong tuong quan > {corr_threshold})...")
        corr_with_target = X.corrwith(y).abs()
        leaky_cols = corr_with_target[corr_with_target > corr_threshold].index.tolist()
        if leaky_cols:
            print(f"[WARN] Loai bo {len(leaky_cols)} cot nghi ngo Data Leakage: {leaky_cols}")
            X = X.drop(columns=leaky_cols)
        else:
            print(f"[INFO] Khong phat hien cot nao co tuong quan > {corr_threshold}")

    X, _ = impute_and_scale(X)
    X_res, y_res = balance_classes(X, y, strategy=balance_strategy)

    out_X = os.path.join(OUTPUT_DIR, "X_train_processed.parquet")
    out_y = os.path.join(OUTPUT_DIR, "y_train_processed.parquet")
    X_res.to_parquet(out_X, index=False)
    y_res.to_frame().to_parquet(out_y, index=False)

    print(f"[DONE] X: {out_X} {X_res.shape} | y: {out_y} {y_res.shape} | So features: {X_res.shape[1]}")
    return X_res, y_res, X_res.columns.tolist()


# entry point
if __name__ == "__main__":
    X, y, features = run_preprocessing(
        balance_strategy="combined",
        drop_leakage=True,
        corr_threshold=0.9,
    )
    print(f"[SUMMARY] X shape: {X.shape} | y distribution:\n{y.value_counts()}")

    # luu cau hinh cho file inference (05)
    model_export_path = r'D:\R STUdio\BHP\03_DL\output\models'
    os.makedirs(model_export_path, exist_ok=True)

    # fit lai scaler tren tap X cuoi cung de dam bao dong bo voi file 05
    scaler_final = StandardScaler()
    scaler_final.fit(X)
    joblib.dump(scaler_final, os.path.join(model_export_path, 'scaler_master.pkl'))
    print(f"[INFO] Da luu Scaler: {os.path.join(model_export_path, 'scaler_master.pkl')}")

    # luu danh sach feature columns (MLP yeu cau dung so luong dau vao)
    joblib.dump(features, os.path.join(model_export_path, 'feature_columns.pkl'))
    print(f"[INFO] Da luu Feature List: {os.path.join(model_export_path, 'feature_columns.pkl')}")
