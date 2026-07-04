r"""Funcoes compartilhadas entre os scripts E*.py e o notebook final
consolidado (04_pipeline_completo_final.ipynb).

Extraido por duplicacao real encontrada nos scripts E*.py (ver
desenvolvimento/docs/plano_tabelas_e_notebook_final.md, Parte B). Nao
adiciona logica nova - apenas centraliza o que ja existia identico ou
quase identico em cada script, para o notebook final nao duplicar.

Constantes e hiperparametros sao reuso do tuning de Barbosa Costa 2026
(nb02 cell 19), documentado no Apendice A do TCC.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
THRESHOLD = 0.09
N_ESTIMATORS = 2000

BEST_LGBM_PARAMS = {
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
}

NB1_DIR = Path("outputs/notebook1")
MAPEAMENTO_PATH = Path("data/anon_outputs/mapeamento_variaveis_inicial.csv")

DROP_HIGH_CARD_TEXT = [
    "d.nom_centro_assist_fam",
    "d.nom_estab_assist_saude_fam",
    "p.nom_escola_memb",
    "p.nom_ibge_munic_nasc_pessoa",
    "p.nom_munic_certid_pessoa",
    "p.nom_munic_escola_memb",
]


def in_colab():
    return "google.colab" in sys.modules


def get_base_path(colab_dir="/content/drive/MyDrive/tcc_cadunico"):
    """Resolve o diretorio base de dados/outputs, Colab ou local.

    No Colab, monta o Drive (se ainda nao montado) e usa `colab_dir`
    como raiz; localmente usa o diretorio de trabalho atual
    (`desenvolvimento/`), inalterado em relacao aos scripts E*.py.
    """
    if in_colab():
        from google.colab import drive  # type: ignore

        if not Path("/content/drive/MyDrive").exists():
            drive.mount("/content/drive")
        base = Path(colab_dir)
        base.mkdir(parents=True, exist_ok=True)
        return base
    return Path(".")


def load_ter_adm_vars(path=MAPEAMENTO_PATH):
    """Fonte unica de TER_VARS/ADM_VARS (E3/E8), via mapeamento_variaveis_inicial.csv.

    Declara 14 TER + 14 ADM. As 14 TER sobrevivem integralmente ao
    pre-processamento; das 14 ADM, apenas 10 sobrevivem (4 identificadores
    indigena/quilombola sao descartados pelo criterio geral de missing
    >= 95% - ver Apendice A, tab:ap_adm). Quem chama esta funcao sobre
    um schema ja preparado deve interseccionar o retorno com as colunas
    efetivamente presentes (ver E3/E8 - ter_in/adm_in = TER_VARS & schema).
    """
    mapeamento = pd.read_csv(path)
    ter_vars = set(mapeamento[mapeamento["grupo"] == "TER"]["coluna"])
    adm_vars = set(mapeamento[mapeamento["grupo"] == "ADM"]["coluna"])
    return ter_vars, adm_vars


def load_notebook1_artifacts(nb1_dir=NB1_DIR):
    """Carrega os pkls do notebook 1 usados por quase todo E*.py."""
    nb1_dir = Path(nb1_dir)
    return {
        "X_train": joblib.load(nb1_dir / "X_train_clipped.pkl"),
        "X_val": joblib.load(nb1_dir / "X_val_clipped.pkl"),
        "y_train": joblib.load(nb1_dir / "y_train.pkl"),
        "y_val": joblib.load(nb1_dir / "y_val.pkl"),
        "X_trainval": joblib.load(nb1_dir / "X_trainval_prepared.pkl"),
        "X_test": joblib.load(nb1_dir / "X_test_prepared.pkl"),
        "y_trainval": joblib.load(nb1_dir / "y_trainval.pkl"),
        "y_test": joblib.load(nb1_dir / "y_test.pkl"),
        "rare_maps": joblib.load(nb1_dir / "rare_maps.pkl"),
        "clip_bounds": joblib.load(nb1_dir / "clip_bounds.pkl"),
    }


def save_notebook1_artifacts(artifacts, nb1_dir=NB1_DIR):
    """Grava os artefatos no mesmo layout esperado por load_notebook1_artifacts."""
    nb1_dir = Path(nb1_dir)
    nb1_dir.mkdir(parents=True, exist_ok=True)
    name_map = {
        "X_train": "X_train_clipped.pkl",
        "X_val": "X_val_clipped.pkl",
        "y_train": "y_train.pkl",
        "y_val": "y_val.pkl",
        "X_trainval": "X_trainval_prepared.pkl",
        "X_test": "X_test_prepared.pkl",
        "y_trainval": "y_trainval.pkl",
        "y_test": "y_test.pkl",
        "rare_maps": "rare_maps.pkl",
        "clip_bounds": "clip_bounds.pkl",
    }
    for key, filename in name_map.items():
        joblib.dump(artifacts[key], nb1_dir / filename)


TARGET = "inseg_alim_bin"
TRAIN_YEAR = 2024
TEST_YEAR = 2025
MISSING_FLAG_THRESHOLD = 0.05
HIGH_MISSING_DROP_THRESHOLD = 0.95
RARE_CATEGORY_THRESHOLD = 0.005
CLIP_LOWER = 0.01
CLIP_UPPER = 0.99
LEAKAGE_COLS = ["d.ind_risco_scl_inseg_alim", "pred_prob_risk", "pred_class_risk"]
POSSIBLE_ID_COLS = ["id_familia", "id_pessoa", "id", "cpf", "nis"]


def _add_date_parts(df_in):
    """Cria atributos derivados de colunas de data (01_preprocess cell 16)."""
    df_out = df_in.copy()
    date_keywords = ["data", "dt_", "_dt", "date", "atualizacao", "cadastro", "entrevista"]
    candidate_cols = [c for c in df_out.columns if any(k in c.lower() for k in date_keywords)]
    for col in candidate_cols:
        try:
            dt = pd.to_datetime(df_out[col], errors="coerce")
            if dt.notna().sum() == 0:
                continue
            df_out[f"{col}_ano"] = dt.dt.year
            df_out[f"{col}_mes"] = dt.dt.month
            df_out[f"{col}_dia"] = dt.dt.day
            df_out[f"{col}_dia_semana"] = dt.dt.dayofweek
            df_out[f"{col}_eh_fim_mes"] = dt.dt.is_month_end.astype("float")
            df_out[f"{col}_eh_inicio_mes"] = dt.dt.is_month_start.astype("float")
        except Exception:
            pass
    return df_out


def _add_missing_flags(df_in, threshold=MISSING_FLAG_THRESHOLD):
    df_out = df_in.copy()
    missing_rate = df_out.isna().mean()
    cols_flag = missing_rate[missing_rate >= threshold].index.tolist()
    for col in cols_flag:
        df_out[f"{col}_is_missing"] = df_out[col].isna().astype(int)
    return df_out


def apply_feature_engineering(df_in):
    """Engenharia de atributos do pre-processamento original (01_preprocess cell 16):
    feat_densidade_domiciliar, feat_proporcao_dependentes, feat_banheiro_por_morador,
    feat_renda_per_capita, feat_pressao_alimentar (Barbosa2026EN secao 3.2)."""
    eps = 1e-6
    df_out = df_in.copy()
    df_out = _add_date_parts(df_out)
    df_out = _add_missing_flags(df_out)
    if {"d.qtd_pessoas_domic_fam", "d.qtd_comodos_dormitorio_fam"}.issubset(df_out.columns):
        df_out["feat_densidade_domiciliar"] = (
            df_out["d.qtd_pessoas_domic_fam"]
            / (df_out["d.qtd_comodos_dormitorio_fam"].replace(0, np.nan) + eps)
        )
    if {"d.qtd_pessoa_inter_0_17_anos_fam", "d.qtd_pessoa_inter_65_anos_fam",
        "d.qtd_pessoas_domic_fam"}.issubset(df_out.columns):
        df_out["feat_proporcao_dependentes"] = (
            df_out["d.qtd_pessoa_inter_0_17_anos_fam"].fillna(0)
            + df_out["d.qtd_pessoa_inter_65_anos_fam"].fillna(0)
        ) / (df_out["d.qtd_pessoas_domic_fam"].replace(0, np.nan) + eps)
    if {"d.qtd_banheiros_fam", "d.qtd_pessoas_domic_fam"}.issubset(df_out.columns):
        df_out["feat_banheiro_por_morador"] = (
            df_out["d.qtd_banheiros_fam"]
            / (df_out["d.qtd_pessoas_domic_fam"].replace(0, np.nan) + eps)
        )
    if {"d.vlr_renda_total_fam", "d.qtd_pessoas_domic_fam"}.issubset(df_out.columns):
        df_out["feat_renda_per_capita"] = (
            df_out["d.vlr_renda_total_fam"]
            / (df_out["d.qtd_pessoas_domic_fam"].replace(0, np.nan) + eps)
        )
    if {"d.val_desp_alimentacao_fam", "d.vlr_renda_total_fam"}.issubset(df_out.columns):
        df_out["feat_pressao_alimentar"] = (
            df_out["d.val_desp_alimentacao_fam"]
            / (df_out["d.vlr_renda_total_fam"].replace(0, np.nan) + eps)
        )
    for col in ["d.vlr_renda_total_fam", "d.vlr_renda_media_fam"]:
        if col in df_out.columns:
            df_out[f"{col}_log1p"] = np.log1p(pd.to_numeric(df_out[col], errors="coerce").clip(lower=0))
    return df_out


def _prepare_types_for_models(train_df, test_df, low_cardinality_threshold=20, drop_high_missing=True):
    """Casting de tipos para modelos tabulares (01_preprocess cell 16)."""
    train_df = train_df.copy()
    test_df = test_df.copy()
    common_cols = [c for c in train_df.columns if c in test_df.columns]
    train_df, test_df = train_df[common_cols].copy(), test_df[common_cols].copy()

    explicit_date_cols = [c for c in [
        "d.dat_atual_fam", "d.dat_cadastramento_fam", "d.dta_entrevista_fam", "p.dta_nasc_pessoa",
    ] if c in train_df.columns and c in test_df.columns]

    for col in explicit_date_cols:
        train_dt = pd.to_datetime(train_df[col], errors="coerce")
        test_dt = pd.to_datetime(test_df[col], errors="coerce")
        for df_, dt_ in [(train_df, train_dt), (test_df, test_dt)]:
            df_[f"{col}_ano"] = dt_.dt.year
            df_[f"{col}_mes"] = dt_.dt.month
            df_[f"{col}_dia"] = dt_.dt.day
            df_[f"{col}_dia_semana"] = dt_.dt.dayofweek
        if col == "p.dta_nasc_pessoa":
            ref_train = pd.Timestamp(f"{TRAIN_YEAR}-12-31")
            ref_test = pd.Timestamp(f"{TEST_YEAR}-12-31")
            train_df["p.idade_aprox"] = (ref_train - train_dt).dt.days / 365.25
            test_df["p.idade_aprox"] = (ref_test - test_dt).dt.days / 365.25

    train_df = train_df.drop(columns=explicit_date_cols, errors="ignore")
    test_df = test_df.drop(columns=explicit_date_cols, errors="ignore")
    common_cols = [c for c in train_df.columns if c in test_df.columns]

    for col in common_cols:
        if train_df[col].dtype == "object" or pd.api.types.is_string_dtype(train_df[col]):
            train_num = pd.to_numeric(train_df[col], errors="coerce")
            test_num = pd.to_numeric(test_df[col], errors="coerce")
            n_orig = train_df[col].notna().sum()
            n_conv = train_num.notna().sum()
            conversion_rate = n_conv / n_orig if n_orig > 0 else 0
            if conversion_rate >= 0.95:
                train_df[col], test_df[col] = train_num, test_num
            else:
                train_df[col] = train_df[col].fillna("MISSING").astype(str).astype("category")
                test_df[col] = test_df[col].fillna("MISSING").astype(str).astype("category")

    common_cols = [c for c in train_df.columns if c in test_df.columns]
    for col in common_cols:
        if pd.api.types.is_numeric_dtype(train_df[col]):
            nunique = train_df[col].nunique(dropna=True)
            cl = col.lower()
            looks_categorical = (
                nunique <= low_cardinality_threshold
                or cl.startswith("cod_") or ".cod_" in cl
                or cl.startswith("ind_") or ".ind_" in cl
                or cl.startswith("fx_") or ".fx_" in cl
                or cl.startswith("sig_") or ".sig_" in cl
            )
            looks_count = "qtd_" in cl or ".qtd_" in cl or "vlr_" in cl or ".vlr_" in cl or "idade" in cl
            if looks_categorical and not looks_count:
                try:
                    train_df[col] = train_df[col].astype("Int64").astype("category")
                    test_df[col] = test_df[col].astype("Int64").astype("category")
                except Exception:
                    pass

    if drop_high_missing:
        missing_rate = train_df.isna().mean()
        cols_to_drop = missing_rate[missing_rate >= HIGH_MISSING_DROP_THRESHOLD].index.tolist()
        if cols_to_drop:
            train_df = train_df.drop(columns=cols_to_drop, errors="ignore")
            test_df = test_df.drop(columns=cols_to_drop, errors="ignore")

    return train_df, test_df


def _group_rare_categories(train_df, other_dfs, min_freq=RARE_CATEGORY_THRESHOLD, rare_label="__RARE__"):
    train_df = train_df.copy()
    transformed_others = [d.copy() for d in other_dfs]
    cat_cols = train_df.select_dtypes(include=["category"]).columns.tolist()
    rare_maps = {}
    for col in cat_cols:
        freq = train_df[col].astype(str).value_counts(normalize=True, dropna=False)
        rare_values = freq[freq < min_freq].index.tolist()
        rare_maps[col] = set(rare_values)
        train_series = train_df[col].astype(str)
        train_df[col] = train_series.where(~train_series.isin(rare_values), rare_label).astype("category")
        for i, df_other in enumerate(transformed_others):
            if col in df_other.columns:
                s = df_other[col].astype(str)
                transformed_others[i][col] = s.where(~s.isin(rare_values), rare_label).astype("category")
    return train_df, transformed_others, rare_maps


def _clip_outliers_by_train(train_df, other_df_list, lower=CLIP_LOWER, upper=CLIP_UPPER):
    train_df = train_df.copy()
    transformed_others = [d.copy() for d in other_df_list]
    num_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    safe_num_cols = []
    for col in num_cols:
        cl = col.lower()
        is_code_like = (
            cl.startswith("cod_") or ".cod_" in cl or cl.startswith("ind_") or ".ind_" in cl
            or cl.startswith("fx_") or ".fx_" in cl or cl.startswith("sig_") or ".sig_" in cl
        )
        low_cardinality = train_df[col].nunique(dropna=True) <= 10
        if not is_code_like and not low_cardinality and col != TARGET:
            safe_num_cols.append(col)
    bounds = {}
    for col in safe_num_cols:
        lo, hi = train_df[col].quantile(lower), train_df[col].quantile(upper)
        bounds[col] = (lo, hi)
        train_df[col] = train_df[col].clip(lo, hi)
    for i, df_other in enumerate(transformed_others):
        for col, (lo, hi) in bounds.items():
            if col in df_other.columns:
                df_other[col] = df_other[col].clip(lo, hi)
        transformed_others[i] = df_other
    return train_df, transformed_others, bounds


def preprocess_union_csv(csv_path, save_to=None):
    """Pipeline completo de pre-processamento a partir do CSV anonimizado
    unido (`cadunico_2023_2025_union_anon.csv`), porta fiel das celulas
    5-21/24-27 de `01_preprocess_and_baselines.ipynb` (filtro de alvo
    valido, split temporal 2024/2025, feature engineering, remocao de
    leakage, casting de tipos, split treino/val, agrupamento de
    categorias raras, clipping de outliers).

    Nao reimplementa a harmonizacao/anonimizacao a partir dos CSVs
    brutos com PII (essa etapa, `limpezaDados.ipynb`, e' codigo da
    co-orientadora e nao e' reexecutada aqui - ver celula comentada na
    Parte 1 do notebook final). `csv_path` deve apontar para o CSV ja
    anonimizado e harmonizado.

    Retorna o mesmo dict de `load_notebook1_artifacts`. Se `save_to`
    for informado, grava os pickles nesse diretorio (cache para
    reexecucoes futuras via `load_notebook1_artifacts`).
    """
    df = pd.read_csv(csv_path)

    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df[np.isfinite(df[TARGET])]
    df = df[df[TARGET].isin([0, 1])].copy()
    df[TARGET] = df[TARGET].astype(int)

    df_trainval = df[df["ano"] == TRAIN_YEAR].copy()
    df_test = df[df["ano"] == TEST_YEAR].copy()

    df_trainval = apply_feature_engineering(df_trainval)
    df_test = apply_feature_engineering(df_test)

    drop_cols = [TARGET, "ano"] + LEAKAGE_COLS
    drop_cols.extend([c for c in POSSIBLE_ID_COLS if c in df_trainval.columns])
    drop_cols = list(dict.fromkeys([c for c in drop_cols if c in df_trainval.columns]))

    X_trainval = df_trainval.drop(columns=drop_cols, errors="ignore").copy()
    y_trainval = df_trainval[TARGET].copy()
    X_test = df_test.drop(columns=drop_cols, errors="ignore").copy()
    y_test = df_test[TARGET].copy()

    X_trainval_prepared, X_test_prepared = _prepare_types_for_models(X_trainval, X_test)

    common_cols = [c for c in X_trainval_prepared.columns if c in X_test_prepared.columns]
    X_trainval_prepared = X_trainval_prepared[common_cols].copy()
    X_test_prepared = X_test_prepared[common_cols].copy()

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval_prepared, y_trainval, test_size=0.2,
        stratify=y_trainval, random_state=RANDOM_STATE,
    )

    X_train_rare, transformed_rare, rare_maps = _group_rare_categories(X_train, [X_val, X_test_prepared])
    X_val_rare, X_test_rare = transformed_rare

    X_train_clipped, transformed_clipped, clip_bounds = _clip_outliers_by_train(
        X_train_rare, [X_val_rare, X_test_rare]
    )
    X_val_clipped, X_test_clipped = transformed_clipped

    artifacts = {
        "X_train": X_train_clipped, "X_val": X_val_clipped,
        "y_train": y_train, "y_val": y_val,
        "X_trainval": X_trainval_prepared, "X_test": X_test_prepared,
        "y_trainval": y_trainval, "y_test": y_test,
        "rare_maps": rare_maps, "clip_bounds": clip_bounds,
    }
    if save_to is not None:
        save_notebook1_artifacts(artifacts, save_to)
    return artifacts


def prepare_lgbm_input(X, reference_columns=None):
    """Converte string/object para Categorical (nb02 cell 23)."""
    X = X.copy()
    cat_cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    for c in cat_cols:
        X[c] = X[c].astype("string")
    for c in cat_cols:
        train_cats = pd.Index(X[c].dropna().unique().tolist()).unique()
        X[c] = pd.Categorical(X[c], categories=train_cats)
    if reference_columns is not None:
        X = X.reindex(columns=reference_columns)
    return X


def cast_numeric_light(X):
    """Reduz memoria: float64->float32, int64->int32 (nb02 cell 18 patched)."""
    X = X.copy()
    for col in X.select_dtypes(include=["float64"]).columns:
        X[col] = X[col].astype("float32")
    for col in X.select_dtypes(include=["int64"]).columns:
        X[col] = X[col].astype("int32")
    return X


def evaluate(y_true, y_proba, threshold=THRESHOLD):
    """Bloco padrao de metricas usado em E1-E3, E8, E10, E11, E15, E16."""
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "AUC-ROC": roc_auc_score(y_true, y_proba),
        "AUC-PR": average_precision_score(y_true, y_proba),
        "F2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Brier": brier_score_loss(y_true, y_proba),
    }


def train_lgbm(X_tr, y_tr, params=None, n_estimators=N_ESTIMATORS,
               random_state=RANDOM_STATE, scale_pos_weight=1.0, categorical_feature=None):
    """Treina um LGBMClassifier com os hiperparametros padrao do trabalho.

    X_tr deve ja estar preparado (prepare_lgbm_input + cast_numeric_light).
    """
    params = params or BEST_LGBM_PARAMS
    if categorical_feature is None:
        categorical_feature = X_tr.select_dtypes(include=["category"]).columns.tolist()
    model = LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1,
        scale_pos_weight=scale_pos_weight,
        **params,
    )
    model.fit(X_tr, y_tr, categorical_feature=categorical_feature)
    return model


def apply_rare_maps(df, rare_maps_dict, rare_label="__RARE__"):
    """Copia exata do nb02 cell 5."""
    df = df.copy()
    for col, rare_values in rare_maps_dict.items():
        if col in df.columns:
            s = df[col].astype(str)
            s = s.where(~s.isin(rare_values), rare_label)
            df[col] = s.astype("category")
    return df


def apply_clip_bounds(df, bounds):
    """Copia exata do nb02 cell 5."""
    df = df.copy()
    for col, (lo, hi) in bounds.items():
        if col in df.columns:
            df[col] = df[col].clip(lo, hi)
    return df


def aggregate_municipal(df, muni_col, y_true_col, y_proba_col, min_obs=30):
    """Agrega predicoes domiciliares em risco municipal (E5/Parte 3.2).

    Retorna DataFrame com n_obs, prev_obs, risco_pred, erro, abs_erro
    por municipio, filtrado a min_obs observacoes.
    """
    mun = df.groupby(muni_col).agg(
        n_obs=(y_true_col, "size"),
        prev_obs=(y_true_col, "mean"),
        risco_pred=(y_proba_col, "mean"),
    ).reset_index()
    mun = mun[mun["n_obs"] >= min_obs].copy()
    mun["erro"] = mun["risco_pred"] - mun["prev_obs"]
    mun["abs_erro"] = mun["erro"].abs()
    return mun
