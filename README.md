# Pipeline de Predição de Insegurança Alimentar em Dados Administrativos em Evolução

Repositório público do Trabalho de Conclusão de Curso (IC/UFAL, 2026).  
**Autor:** Vitor Magno Gouveia  
**Orientadora:** Keila Barbosa Costa  
**Instituição:** Instituto de Computação — Universidade Federal de Alagoas (IC/UFAL)

---

## Descrição

Este repositório contém o pipeline de predição de insegurança alimentar desenvolvido
como TCC. O trabalho avalia empiricamente a robustez espaço-temporal de um modelo
LightGBM treinado sobre microdados anonimizados do CadÚnico em Alagoas (ciclos 2024
e 2025), decompondo a contribuição de variáveis territoriais diretas (TER) e de
variáveis administrativas territorialmente dependentes (ADM).

---

## Reprodução

### 1. Instalar dependências

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.\.venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

### 2. Obter os dados

Os microdados anonimizados do CadÚnico não são distribuídos diretamente neste
repositório por restrições de privacidade (LGPD). Para acesso, consulte o repositório
da orientadora ou entre em contato com os autores.

Coloque o arquivo `cadunico_2023_2025_union_anon.csv` em `data/anon_outputs/`.

### 3. Executar o pipeline

Abra e execute `04_pipeline_completo_final.ipynb` na ordem das células.

O notebook detecta automaticamente se os artefatos de pré-processamento em cache
(`outputs/notebook1/`) já existem; se não existirem, executa o pré-processamento
completo a partir do CSV.

---

## Estrutura do repositório

| Arquivo/Pasta | Descrição |
|---|---|
| `04_pipeline_completo_final.ipynb` | Pipeline completo auto-suficiente (Experimentos E1–E12) |
| `scripts/common_pipeline.py` | Módulo auxiliar compartilhado (pré-processamento, treino, métricas) |
| `requirements.txt` | Dependências Python com versões fixadas |
| `data/BR_Municipios_2024/` | Shapefile IBGE de municípios brasileiros (dado público) |
| `data/anon_outputs/mapeamento_variaveis_inicial.csv` | Classificação TER/ADM/SOC/TEMP/OTHER das variáveis |

---

## Referências

BARBOSA COSTA, Keila et al. A Machine Learning Framework for Early Detection of Food
Insecurity Using Administrative Microdata. In: *Anais do CSBC 2026 — LASDigiGov*.
Sociedade Brasileira de Computação, 2026.
Código: <https://github.com/keilabcs/CadUnicoIA>

BARBOSA, Keila; DAMIÃO, Gabriel; MENDES, Wictória; AQUINO, André L. Harmonização
Longitudinal e Dinâmica Espacial da Insegurança Alimentar em Microdados do CadÚnico.
In: *Anais do CSBC 2026 — LASDigiGov*. Sociedade Brasileira de Computação, 2026.
Código: <https://github.com/keilabcs/CadUnico>

CHRISTENSEN, C.; WAGNER, T.; LANGHALS, B. Year-Independent Prediction of Food
Insecurity Using Classical and Neural Network Machine Learning Methods. *AI*, v. 2,
p. 244–260, 2021.

MORAIS, Dayane de Castro et al. Indicadores de avaliação da Insegurança Alimentar e
Nutricional e fatores associados: revisão sistemática. *Ciência & Saúde Coletiva*, 2020.
