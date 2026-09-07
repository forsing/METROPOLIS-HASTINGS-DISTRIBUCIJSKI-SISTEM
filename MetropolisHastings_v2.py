#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LOTO 7/39 — METROPOLIS–HASTINGS DISTRIBUCIJSKI SISTEM

Jedno konačno rešenje:

1. distribucijski gradient-boosting regresor;
2. grafovski skor parova;
3. Metropolis–Hastings izbor kombinacije;
4. hronološka validacija;
5. potpuno zamrznuti holdout;
6. blok-bootstrap interval od 95%;
7. Monte Karlo test prema poštenom Loto 7/39 procesu.

Sva istorijska izvlačenja učitavaju se iz jednog zajedničkog CSV fajla.
Sistem daje jednu NEXT kombinaciju.

Prvi CSV red predstavlja najstarije izvlačenje.
Poslednji CSV red predstavlja najnovije izvlačenje.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error


# =============================================================================
# PODEŠAVANJA
# =============================================================================

SEED = 39

ZAJEDNICKI_CSV = Path(
    "/data/loto7_4680_k71.csv"
)

BROJ_KUGLICA = 39
BROJ_IZVUCENIH = 7

TEORIJSKA_STOPA = BROJ_IZVUCENIH / BROJ_KUGLICA
SLUCAJNO_OCEKIVANJE = BROJ_IZVUCENIH**2 / BROJ_KUGLICA
UKUPNO_KOMBINACIJA = math.comb(BROJ_KUGLICA, BROJ_IZVUCENIH)

MINIMALNA_ISTORIJA = 250

HORIZONT_METE = 5
TEZINE_METE = np.array(
    [1.00, 0.75, 0.55, 0.40, 0.30],
    dtype=float,
)
TEZINE_METE /= TEZINE_METE.sum()

BROJ_VALIDACIONIH_IZVLACENJA = 120
BROJ_HOLDOUT_IZVLACENJA = 120

MAKSIMALNO_TRENING_TRENUTAKA_TUNING = 700
MAKSIMALNO_TRENING_TRENUTAKA_HOLDOUT = 800
MAKSIMALNO_TRENING_TRENUTAKA_NEXT = 900

PROZORI = (20, 50, 100, 200)
PROZOR_EWMA = 400
PROZOR_TRANZICIJE = 300
PROZOR_GRAFA = 250

LAMBDA_KANDIDATI = (
    0.00,
    0.10,
    0.25,
    0.50,
    1.00,
)

TEMPERATURA = 0.75

MH_VALIDACIONI_LANCI = 3
MH_VALIDACIONI_KORACI = 1_500
MH_VALIDACIONI_BURN_IN = 300

MH_HOLDOUT_LANCI = 3
MH_HOLDOUT_KORACI = 2_000
MH_HOLDOUT_BURN_IN = 400

MH_NEXT_LANCI = 32
MH_NEXT_KORACI = 100_000
MH_NEXT_BURN_IN = 20_000

BROJ_BOOTSTRAP_PONAVLJANJA = 10_000
BROJ_MONTE_KARLO_PONAVLJANJA = 100_000
BOOTSTRAP_BLOK = 12
NIVO_ZNACAJNOSTI = 0.05

warnings.filterwarnings("ignore")


# =============================================================================
# KONFIGURACIJE REGRESORA
# =============================================================================

KONFIGURACIJE_MODELA = (
    {
        "learning_rate": 0.035,
        "max_iter": 300,
        "max_leaf_nodes": 15,
        "max_depth": None,
        "min_samples_leaf": 30,
        "l2_regularization": 1.0,
    },
    {
        "learning_rate": 0.025,
        "max_iter": 400,
        "max_leaf_nodes": 31,
        "max_depth": None,
        "min_samples_leaf": 40,
        "l2_regularization": 2.0,
    },
    {
        "learning_rate": 0.050,
        "max_iter": 250,
        "max_leaf_nodes": 15,
        "max_depth": 6,
        "min_samples_leaf": 50,
        "l2_regularization": 3.0,
    },
)


# =============================================================================
# REZULTAT
# =============================================================================

@dataclass
class Rezultat:
    naziv: str
    broj_redova: int
    next_kombinacija: list[int]
    next_skor: float
    prihvatanje_mh: float
    izabrana_lambda: float
    izabrana_konfiguracija: int
    validacioni_prosek: float
    holdout_mae: float
    holdout_prosek: float
    holdout_medijana: float
    holdout_maksimum: int
    donja_granica_95: float
    gornja_granica_95: float
    p_vrednost: float
    statisticki_pouzdano: bool
    broj_validacionih_izvlacenja: int
    broj_holdout_izvlacenja: int


# =============================================================================
# UČITAVANJE
# =============================================================================

def ucitaj_csv(putanja: Path) -> np.ndarray:
    if not putanja.exists():
        raise FileNotFoundError(
            f"CSV fajl ne postoji: {putanja}"
        )

    okvir = pd.read_csv(
        putanja,
        header=None,
    )

    okvir = okvir.apply(
        pd.to_numeric,
        errors="coerce",
    )

    okvir = okvir.dropna(
        axis=0,
        how="all",
    )

    if okvir.shape[1] < BROJ_IZVUCENIH:
        raise ValueError(
            "CSV mora imati najmanje sedam kolona."
        )

    okvir = okvir.iloc[:, :BROJ_IZVUCENIH]

    if okvir.isna().any().any():
        raise ValueError(
            "CSV sadrži vrednosti koje nisu brojevi."
        )

    podaci = okvir.to_numpy(dtype=int)

    potreban_broj = (
        MINIMALNA_ISTORIJA
        + BROJ_VALIDACIONIH_IZVLACENJA
        + BROJ_HOLDOUT_IZVLACENJA
        + HORIZONT_METE
    )

    if len(podaci) < potreban_broj:
        raise ValueError(
            f"Potrebno je najmanje {potreban_broj} izvlačenja."
        )

    for redni_broj, red in enumerate(
        podaci,
        start=1,
    ):
        if len(np.unique(red)) != BROJ_IZVUCENIH:
            raise ValueError(
                f"Red {redni_broj} nema sedam različitih brojeva."
            )

        if np.any(red < 1) or np.any(
            red > BROJ_KUGLICA
        ):
            raise ValueError(
                f"Red {redni_broj} sadrži broj izvan opsega 1–39."
            )

    return podaci


def napravi_binarnu_matricu(
    izvlacenja: np.ndarray,
) -> np.ndarray:
    binarna = np.zeros(
        (len(izvlacenja), BROJ_KUGLICA),
        dtype=np.float64,
    )

    for t, red in enumerate(izvlacenja):
        binarna[t, red - 1] = 1.0

    return binarna


# =============================================================================
# POMOĆNE FUNKCIJE
# =============================================================================

def standardizuj_vektor(
    vrednosti: np.ndarray,
) -> np.ndarray:
    vrednosti = np.asarray(
        vrednosti,
        dtype=float,
    )

    sredina = float(
        np.mean(vrednosti)
    )

    odstupanje = float(
        np.std(vrednosti)
    )

    if odstupanje < 1e-12:
        return np.zeros_like(
            vrednosti
        )

    return (
        vrednosti - sredina
    ) / odstupanje


def ravnomerni_trenuci(
    pocetak: int,
    kraj: int,
    maksimum: int,
) -> np.ndarray:
    if kraj <= pocetak:
        raise ValueError(
            "Neispravan hronološki opseg."
        )

    broj = min(
        maksimum,
        kraj - pocetak,
    )

    return np.unique(
        np.linspace(
            pocetak,
            kraj - 1,
            num=broj,
            dtype=int,
        )
    )


# =============================================================================
# DISTRIBUCIJSKE OSOBINE
# =============================================================================

def rolling_odnos(
    istorija: np.ndarray,
    prozor: int | None,
) -> np.ndarray:
    deo = (
        istorija
        if prozor is None
        else istorija[-prozor:]
    )

    if len(deo) == 0:
        return np.ones(
            BROJ_KUGLICA,
            dtype=float,
        )

    c = deo.sum(axis=0)
    E = len(deo) * TEORIJSKA_STOPA

    return (
        c + E
    ) / (
        2.0 * E
    )


def ewma_stopa(
    istorija: np.ndarray,
    poluzivot: float,
) -> np.ndarray:
    deo = istorija[
        -PROZOR_EWMA:
    ]

    n = len(deo)

    if n == 0:
        return np.full(
            BROJ_KUGLICA,
            TEORIJSKA_STOPA,
            dtype=float,
        )

    starost = np.arange(
        n - 1,
        -1,
        -1,
        dtype=float,
    )

    tezine = np.power(
        0.5,
        starost / poluzivot,
    )

    return (
        tezine @ deo
    ) / tezine.sum()


def gap_i_hazard(
    istorija: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    trenutni_gap = np.zeros(
        BROJ_KUGLICA,
        dtype=float,
    )

    prosecni_gap = np.zeros(
        BROJ_KUGLICA,
        dtype=float,
    )

    hazard = np.zeros(
        BROJ_KUGLICA,
        dtype=float,
    )

    for broj in range(
        BROJ_KUGLICA
    ):
        pozicije = np.flatnonzero(
            istorija[:, broj] > 0.5
        )

        if len(pozicije) == 0:
            trenutni_gap[broj] = len(
                istorija
            )

            prosecni_gap[broj] = (
                1.0 / TEORIJSKA_STOPA
            )

            hazard[broj] = (
                TEORIJSKA_STOPA
            )

            continue

        trenutni = (
            len(istorija)
            - 1
            - int(pozicije[-1])
        )

        trenutni_gap[broj] = trenutni

        if len(pozicije) < 2:
            prosecni_gap[broj] = (
                1.0 / TEORIJSKA_STOPA
            )

            hazard[broj] = (
                TEORIJSKA_STOPA
            )

            continue

        zavrseni = np.diff(
            pozicije
        )

        prosecni_gap[broj] = float(
            np.mean(zavrseni)
        )

        prag = trenutni + 1

        broj_rizicnih = int(
            np.sum(
                zavrseni >= prag
            )
        )

        broj_dogadjaja = int(
            np.sum(
                zavrseni == prag
            )
        )

        hazard[broj] = (
            broj_dogadjaja + 1.0
        ) / (
            broj_rizicnih + 2.0
        )

    return (
        np.log1p(trenutni_gap),
        np.log1p(prosecni_gap),
        hazard,
    )


def tranzicioni_skor(
    istorija: np.ndarray,
) -> np.ndarray:
    if len(istorija) < 2:
        return np.full(
            BROJ_KUGLICA,
            TEORIJSKA_STOPA,
            dtype=float,
        )

    pocetak = max(
        1,
        len(istorija)
        - PROZOR_TRANZICIJE,
    )

    prethodni_brojevi = np.flatnonzero(
        istorija[-1] > 0.5
    )

    prethodna_stanja = istorija[
        pocetak - 1:-1
    ]

    sledeca_stanja = istorija[
        pocetak:
    ]

    skorovi = []

    for prethodni in prethodni_brojevi:
        maska = (
            prethodna_stanja[
                :,
                prethodni,
            ]
            > 0.5
        )

        broj_uslova = int(
            maska.sum()
        )

        if broj_uslova == 0:
            skorovi.append(
                np.full(
                    BROJ_KUGLICA,
                    TEORIJSKA_STOPA,
                    dtype=float,
                )
            )

            continue

        broj_prelaza = sledeca_stanja[
            maska
        ].sum(axis=0)

        stopa = (
            broj_prelaza
            + broj_uslova
            * TEORIJSKA_STOPA
        ) / (
            2.0 * broj_uslova
        )

        skorovi.append(stopa)

    if not skorovi:
        return np.full(
            BROJ_KUGLICA,
            TEORIJSKA_STOPA,
            dtype=float,
        )

    return np.mean(
        np.asarray(skorovi),
        axis=0,
    )


# =============================================================================
# GRAF PAROVA
# =============================================================================

def grafovska_matrica(
    istorija: np.ndarray,
) -> np.ndarray:
    deo = istorija[
        -PROZOR_GRAFA:
    ]

    n = len(deo)

    if n == 0:
        return np.zeros(
            (
                BROJ_KUGLICA,
                BROJ_KUGLICA,
            ),
            dtype=float,
        )

    broj_pojavljivanja = (
        deo.sum(axis=0)
    )

    zajednicko = (
        deo.T @ deo
    )

    ocekivano = np.outer(
        broj_pojavljivanja,
        broj_pojavljivanja,
    ) / max(float(n), 1.0)

    odnos = (
        zajednicko + ocekivano + 1.0
    ) / (
        2.0 * ocekivano + 1.0
    )

    log_odnos = np.log(
        np.clip(
            odnos,
            1e-8,
            None,
        )
    )

    maska = ~np.eye(
        BROJ_KUGLICA,
        dtype=bool,
    )

    vrednosti = log_odnos[
        maska
    ]

    sredina = float(
        np.mean(vrednosti)
    )

    odstupanje = float(
        np.std(vrednosti)
    )

    if odstupanje > 1e-12:
        graf = (
            log_odnos - sredina
        ) / odstupanje
    else:
        graf = np.zeros_like(
            log_odnos
        )

    graf = (
        graf + graf.T
    ) / 2.0

    np.fill_diagonal(
        graf,
        0.0,
    )

    return np.clip(
        graf,
        -5.0,
        5.0,
    )


def grafovske_osobine(
    istorija: np.ndarray,
    graf: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    poslednji = np.flatnonzero(
        istorija[-1] > 0.5
    )

    prema_poslednjem = np.mean(
        graf[:, poslednji],
        axis=1,
    )

    centralnost = np.mean(
        graf,
        axis=1,
    )

    return (
        prema_poslednjem,
        centralnost,
    )


# =============================================================================
# MATRICA OSOBINA ZA JEDAN TRENUTAK
# =============================================================================

def osobine_za_trenutak(
    binarna: np.ndarray,
    t: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    istorija = binarna[:t]

    odnosi = [
        rolling_odnos(
            istorija,
            prozor,
        )
        for prozor in PROZORI
    ]

    odnos_sve = rolling_odnos(
        istorija,
        None,
    )

    trend_20_100 = (
        odnosi[0] - odnosi[2]
    )

    trend_50_200 = (
        odnosi[1] - odnosi[3]
    )

    ewma_20 = ewma_stopa(
        istorija,
        20.0,
    )

    ewma_60 = ewma_stopa(
        istorija,
        60.0,
    )

    (
        log_trenutni_gap,
        log_prosecni_gap,
        hazard,
    ) = gap_i_hazard(
        istorija
    )

    tranzicija = tranzicioni_skor(
        istorija
    )

    graf = grafovska_matrica(
        istorija
    )

    (
        graf_poslednje,
        graf_centralnost,
    ) = grafovske_osobine(
        istorija,
        graf,
    )

    prethodno = istorija[-1]

    brojevi = np.arange(
        1,
        BROJ_KUGLICA + 1,
        dtype=float,
    )

    ugao = (
        2.0
        * math.pi
        * brojevi
        / BROJ_KUGLICA
    )

    X = np.column_stack(
        [
            odnosi[0],
            odnosi[1],
            odnosi[2],
            odnosi[3],
            odnos_sve,
            trend_20_100,
            trend_50_200,
            ewma_20,
            ewma_60,
            log_trenutni_gap,
            log_prosecni_gap,
            hazard,
            tranzicija,
            graf_poslednje,
            graf_centralnost,
            prethodno,
            np.sin(ugao),
            np.cos(ugao),
        ]
    )

    X = np.nan_to_num(
        X,
        nan=0.0,
        posinf=10.0,
        neginf=-10.0,
    )

    return X, graf


def meta_za_trenutak(
    binarna: np.ndarray,
    t: int,
) -> np.ndarray:
    buducnost = binarna[
        t:t + HORIZONT_METE
    ]

    if len(buducnost) != HORIZONT_METE:
        raise ValueError(
            "Nema dovoljno budućih redova za metu."
        )

    return (
        TEZINE_METE @ buducnost
    )


def napravi_dataset(
    binarna: np.ndarray,
    trenuci: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    X_delovi = []
    y_delovi = []

    for t in trenuci:
        X_t, _ = osobine_za_trenutak(
            binarna,
            int(t),
        )

        y_t = meta_za_trenutak(
            binarna,
            int(t),
        )

        X_delovi.append(X_t)
        y_delovi.append(y_t)

    return (
        np.vstack(X_delovi),
        np.concatenate(y_delovi),
    )


# =============================================================================
# GRADIENT BOOSTING REGRESOR
# =============================================================================

def napravi_model(
    konfiguracija: dict,
) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=konfiguracija[
            "learning_rate"
        ],
        max_iter=konfiguracija[
            "max_iter"
        ],
        max_leaf_nodes=konfiguracija[
            "max_leaf_nodes"
        ],
        max_depth=konfiguracija[
            "max_depth"
        ],
        min_samples_leaf=konfiguracija[
            "min_samples_leaf"
        ],
        l2_regularization=konfiguracija[
            "l2_regularization"
        ],
        early_stopping=False,
        random_state=SEED,
    )


# =============================================================================
# METROPOLIS–HASTINGS
# =============================================================================

def skor_kombinacije(
    kombinacija: np.ndarray,
    skorovi_brojeva: np.ndarray,
    graf: np.ndarray,
    lambda_graf: float,
) -> float:
    pojedinacni = float(
        skorovi_brojeva[
            kombinacija
        ].sum()
    )

    podmatrica = graf[
        np.ix_(
            kombinacija,
            kombinacija,
        )
    ]

    parovi = float(
        np.triu(
            podmatrica,
            k=1,
        ).sum()
    )

    return (
        pojedinacni
        + lambda_graf * parovi
    )


def metropolis_hastings(
    skorovi_brojeva: np.ndarray,
    graf: np.ndarray,
    lambda_graf: float,
    broj_lanaca: int,
    broj_koraka: int,
    burn_in: int,
    seed: int,
) -> tuple[
    list[int],
    float,
    float,
]:
    rng = np.random.default_rng(
        seed
    )

    skorovi = standardizuj_vektor(
        skorovi_brojeva
    )

    najbolja_kombinacija = None
    najbolji_skor = -np.inf

    ukupno_prihvaceno = 0
    ukupno_predlozeno = 0

    svi_brojevi = np.arange(
        BROJ_KUGLICA,
        dtype=int,
    )

    for _ in range(broj_lanaca):
        kombinacija = np.sort(
            rng.choice(
                BROJ_KUGLICA,
                size=BROJ_IZVUCENIH,
                replace=False,
            )
        )

        u_kombinaciji = np.zeros(
            BROJ_KUGLICA,
            dtype=bool,
        )

        u_kombinaciji[
            kombinacija
        ] = True

        trenutni_skor = skor_kombinacije(
            kombinacija,
            skorovi,
            graf,
            lambda_graf,
        )

        for korak in range(
            broj_koraka
        ):
            pozicija_za_zamenu = int(
                rng.integers(
                    0,
                    BROJ_IZVUCENIH,
                )
            )

            uklonjeni = int(
                kombinacija[
                    pozicija_za_zamenu
                ]
            )

            kandidati_za_dodavanje = svi_brojevi[
                ~u_kombinaciji
            ]

            dodati = int(
                rng.choice(
                    kandidati_za_dodavanje
                )
            )

            preostali = kombinacija[
                kombinacija != uklonjeni
            ]

            promena_pojedinacnog = (
                skorovi[dodati]
                - skorovi[uklonjeni]
            )

            promena_parova = (
                graf[
                    dodati,
                    preostali,
                ].sum()
                - graf[
                    uklonjeni,
                    preostali,
                ].sum()
            )

            promena_skora = float(
                promena_pojedinacnog
                + lambda_graf
                * promena_parova
            )

            if promena_skora >= 0.0:
                prihvati = True
            else:
                verovatnoca = math.exp(
                    max(
                        -700.0,
                        promena_skora
                        / TEMPERATURA,
                    )
                )

                prihvati = bool(
                    rng.random()
                    < verovatnoca
                )

            ukupno_predlozeno += 1

            if prihvati:
                ukupno_prihvaceno += 1

                u_kombinaciji[
                    uklonjeni
                ] = False

                u_kombinaciji[
                    dodati
                ] = True

                kombinacija[
                    pozicija_za_zamenu
                ] = dodati

                kombinacija.sort()

                trenutni_skor += (
                    promena_skora
                )

            if (
                korak >= burn_in
                and trenutni_skor
                > najbolji_skor
            ):
                najbolji_skor = float(
                    trenutni_skor
                )

                najbolja_kombinacija = (
                    kombinacija.copy()
                )

    if najbolja_kombinacija is None:
        raise RuntimeError(
            "Metropolis–Hastings nije pronašao kombinaciju."
        )

    stopa_prihvatanja = (
        ukupno_prihvaceno
        / max(
            ukupno_predlozeno,
            1,
        )
    )

    return (
        sorted(
            int(broj + 1)
            for broj
            in najbolja_kombinacija
        ),
        float(najbolji_skor),
        float(stopa_prihvatanja),
    )


# =============================================================================
# VALIDACIJA MODELA I LAMBDA VREDNOSTI
# =============================================================================

def validiraj_kombinaciju(
    model: HistGradientBoostingRegressor,
    binarna: np.ndarray,
    validacioni_trenuci: np.ndarray,
    lambda_graf: float,
    seed_pomeraj: int,
) -> tuple[
    float,
    float,
]:
    pogodci = []
    apsolutne_greske = []

    for redni_broj, t in enumerate(
        validacioni_trenuci
    ):
        X_t, graf = osobine_za_trenutak(
            binarna,
            int(t),
        )

        skorovi = model.predict(
            X_t
        )

        kombinacija, _, _ = metropolis_hastings(
            skorovi_brojeva=skorovi,
            graf=graf,
            lambda_graf=lambda_graf,
            broj_lanaca=MH_VALIDACIONI_LANCI,
            broj_koraka=MH_VALIDACIONI_KORACI,
            burn_in=MH_VALIDACIONI_BURN_IN,
            seed=(
                SEED
                + seed_pomeraj
                + int(t) * 101
                + redni_broj
            ),
        )

        indeksi = np.asarray(
            kombinacija,
            dtype=int,
        ) - 1

        stvarni = binarna[
            int(t)
        ]

        pogodci.append(
            int(
                stvarni[
                    indeksi
                ].sum()
            )
        )

        apsolutne_greske.append(
            float(
                np.mean(
                    np.abs(
                        stvarni - skorovi
                    )
                )
            )
        )

    return (
        float(np.mean(pogodci)),
        float(
            np.mean(
                apsolutne_greske
            )
        ),
    )


def izaberi_model_i_lambda(
    binarna: np.ndarray,
    trening_trenuci: np.ndarray,
    validacioni_trenuci: np.ndarray,
    seed_pomeraj: int,
) -> tuple[
    int,
    float,
    float,
]:
    X_train, y_train = napravi_dataset(
        binarna,
        trening_trenuci,
    )

    najbolji_kljuc = None
    najbolja_konfiguracija = None
    najbolja_lambda = None
    najbolji_prosek = None

    print(
        "Hronološki izbor regresora i "
        "grafovske lambda vrednosti..."
    )

    for indeks_konfiguracije, konfiguracija in enumerate(
        KONFIGURACIJE_MODELA,
        start=1,
    ):
        model = napravi_model(
            konfiguracija
        )

        model.fit(
            X_train,
            y_train,
        )

        for lambda_graf in LAMBDA_KANDIDATI:
            (
                prosek_pogodaka,
                validacioni_mae,
            ) = validiraj_kombinaciju(
                model=model,
                binarna=binarna,
                validacioni_trenuci=validacioni_trenuci,
                lambda_graf=lambda_graf,
                seed_pomeraj=(
                    seed_pomeraj
                    + indeks_konfiguracije
                    * 10_000
                    + int(
                        lambda_graf * 1_000
                    )
                ),
            )

            print(
                f"  Model {indeks_konfiguracije}, "
                f"lambda={lambda_graf:.2f}: "
                f"prosek={prosek_pogodaka:.6f}, "
                f"MAE={validacioni_mae:.6f}"
            )

            kljuc = (
                prosek_pogodaka,
                -validacioni_mae,
                -lambda_graf,
                -indeks_konfiguracije,
            )

            if (
                najbolji_kljuc is None
                or kljuc > najbolji_kljuc
            ):
                najbolji_kljuc = kljuc
                najbolja_konfiguracija = (
                    indeks_konfiguracije
                )
                najbolja_lambda = float(
                    lambda_graf
                )
                najbolji_prosek = float(
                    prosek_pogodaka
                )

    return (
        int(najbolja_konfiguracija),
        float(najbolja_lambda),
        float(najbolji_prosek),
    )


# =============================================================================
# ZAMRZNUTI HOLDOUT
# =============================================================================

def proveri_holdout(
    model: HistGradientBoostingRegressor,
    binarna: np.ndarray,
    holdout_trenuci: np.ndarray,
    lambda_graf: float,
    seed_pomeraj: int,
) -> tuple[
    np.ndarray,
    float,
]:
    pogodci = []
    mae_vrednosti = []

    print(
        "Zamrznuta holdout provera..."
    )

    for redni_broj, t in enumerate(
        holdout_trenuci
    ):
        X_t, graf = osobine_za_trenutak(
            binarna,
            int(t),
        )

        skorovi = model.predict(
            X_t
        )

        kombinacija, _, _ = metropolis_hastings(
            skorovi_brojeva=skorovi,
            graf=graf,
            lambda_graf=lambda_graf,
            broj_lanaca=MH_HOLDOUT_LANCI,
            broj_koraka=MH_HOLDOUT_KORACI,
            burn_in=MH_HOLDOUT_BURN_IN,
            seed=(
                SEED
                + seed_pomeraj
                + int(t) * 1_009
                + redni_broj
            ),
        )

        indeksi = np.asarray(
            kombinacija,
            dtype=int,
        ) - 1

        stvarni = binarna[
            int(t)
        ]

        pogodci.append(
            int(
                stvarni[
                    indeksi
                ].sum()
            )
        )

        mae_vrednosti.append(
            float(
                np.mean(
                    np.abs(
                        stvarni - skorovi
                    )
                )
            )
        )

        if (
            redni_broj == 0
            or (redni_broj + 1) % 20 == 0
            or redni_broj + 1
            == len(holdout_trenuci)
        ):
            print(
                f"  Holdout "
                f"{redni_broj + 1}/"
                f"{len(holdout_trenuci)}"
            )

    return (
        np.asarray(
            pogodci,
            dtype=int,
        ),
        float(
            np.mean(
                mae_vrednosti
            )
        ),
    )


# =============================================================================
# STATISTIČKA PROVERA
# =============================================================================

def blok_bootstrap_interval(
    pogodci: np.ndarray,
    seed: int,
) -> tuple[
    float,
    float,
]:
    pogodci = np.asarray(
        pogodci,
        dtype=float,
    )

    n = len(pogodci)

    rng = np.random.default_rng(
        seed
    )

    velicina_bloka = min(
        BOOTSTRAP_BLOK,
        n,
    )

    broj_blokova = math.ceil(
        n / velicina_bloka
    )

    najveci_pocetak = (
        n - velicina_bloka
    )

    proseci = np.empty(
        BROJ_BOOTSTRAP_PONAVLJANJA,
        dtype=float,
    )

    for ponavljanje in range(
        BROJ_BOOTSTRAP_PONAVLJANJA
    ):
        delovi = []

        for _ in range(
            broj_blokova
        ):
            pocetak = int(
                rng.integers(
                    0,
                    najveci_pocetak + 1,
                )
            )

            delovi.append(
                pogodci[
                    pocetak:
                    pocetak + velicina_bloka
                ]
            )

        uzorak = np.concatenate(
            delovi
        )[:n]

        proseci[
            ponavljanje
        ] = float(
            uzorak.mean()
        )

    donja, gornja = np.quantile(
        proseci,
        [0.025, 0.975],
    )

    return (
        float(donja),
        float(gornja),
    )


def monte_karlo_p_vrednost(
    posmatrani_prosek: float,
    broj_izvlacenja: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(
        seed
    )

    simulirani = rng.hypergeometric(
        ngood=BROJ_IZVUCENIH,
        nbad=(
            BROJ_KUGLICA
            - BROJ_IZVUCENIH
        ),
        nsample=BROJ_IZVUCENIH,
        size=(
            BROJ_MONTE_KARLO_PONAVLJANJA,
            broj_izvlacenja,
        ),
    )

    simulirani_proseci = simulirani.mean(
        axis=1
    )

    broj_boljih = int(
        np.sum(
            simulirani_proseci
            >= posmatrani_prosek
        )
    )

    return float(
        (
            broj_boljih + 1
        )
        / (
            BROJ_MONTE_KARLO_PONAVLJANJA
            + 1
        )
    )


# =============================================================================
# OBRADA JEDNE IGRE
# =============================================================================

def obradi_igru(
    naziv: str,
    putanja: Path,
    seed_pomeraj: int,
) -> Rezultat:
    print()
    print("=" * 78)
    print(f"Obrada: {naziv}")
    print("=" * 78)
    print(f"CSV: {putanja}")

    izvlacenja = ucitaj_csv(
        putanja
    )

    binarna = napravi_binarnu_matricu(
        izvlacenja
    )

    n = len(binarna)

    print(f"Broj redova: {n}")
    print(
        "Prvi red se tretira kao najstariji."
    )
    print(
        "Poslednji red se tretira kao najnoviji."
    )

    holdout_pocetak = (
        n
        - BROJ_HOLDOUT_IZVLACENJA
    )

    validacija_pocetak = (
        holdout_pocetak
        - BROJ_VALIDACIONIH_IZVLACENJA
    )

    tuning_trening_kraj = (
        validacija_pocetak
        - HORIZONT_METE
        + 1
    )

    tuning_trenuci = ravnomerni_trenuci(
        MINIMALNA_ISTORIJA,
        tuning_trening_kraj,
        MAKSIMALNO_TRENING_TRENUTAKA_TUNING,
    )

    validacioni_trenuci = np.arange(
        validacija_pocetak,
        holdout_pocetak,
        dtype=int,
    )

    (
        izabrana_konfiguracija,
        izabrana_lambda,
        validacioni_prosek,
    ) = izaberi_model_i_lambda(
        binarna=binarna,
        trening_trenuci=tuning_trenuci,
        validacioni_trenuci=validacioni_trenuci,
        seed_pomeraj=seed_pomeraj,
    )

    print(
        "Izabrana konfiguracija regresora: "
        f"{izabrana_konfiguracija}"
    )

    print(
        "Izabrana grafovska lambda: "
        f"{izabrana_lambda:.2f}"
    )

    print(
        "Validacioni prosek pogodaka: "
        f"{validacioni_prosek:.6f}"
    )

    holdout_trening_kraj = (
        holdout_pocetak
        - HORIZONT_METE
        + 1
    )

    holdout_trening_trenuci = (
        ravnomerni_trenuci(
            MINIMALNA_ISTORIJA,
            holdout_trening_kraj,
            MAKSIMALNO_TRENING_TRENUTAKA_HOLDOUT,
        )
    )

    X_holdout_train, y_holdout_train = (
        napravi_dataset(
            binarna,
            holdout_trening_trenuci,
        )
    )

    holdout_model = napravi_model(
        KONFIGURACIJE_MODELA[
            izabrana_konfiguracija - 1
        ]
    )

    holdout_model.fit(
        X_holdout_train,
        y_holdout_train,
    )

    holdout_trenuci = np.arange(
        holdout_pocetak,
        n,
        dtype=int,
    )

    (
        holdout_pogodci,
        holdout_mae,
    ) = proveri_holdout(
        model=holdout_model,
        binarna=binarna,
        holdout_trenuci=holdout_trenuci,
        lambda_graf=izabrana_lambda,
        seed_pomeraj=(
            seed_pomeraj + 500_000
        ),
    )

    holdout_prosek = float(
        holdout_pogodci.mean()
    )

    (
        donja_granica_95,
        gornja_granica_95,
    ) = blok_bootstrap_interval(
        holdout_pogodci,
        seed=SEED + seed_pomeraj,
    )

    p_vrednost = monte_karlo_p_vrednost(
        posmatrani_prosek=holdout_prosek,
        broj_izvlacenja=len(
            holdout_pogodci
        ),
        seed=(
            SEED
            + seed_pomeraj
            + 900_000
        ),
    )

    statisticki_pouzdano = bool(
        donja_granica_95
        > SLUCAJNO_OCEKIVANJE
        and p_vrednost
        < NIVO_ZNACAJNOSTI
    )

    print(
        "Završna obuka nad svim dostupnim "
        "poznatim metama..."
    )

    next_trening_kraj = (
        n
        - HORIZONT_METE
        + 1
    )

    next_trening_trenuci = (
        ravnomerni_trenuci(
            MINIMALNA_ISTORIJA,
            next_trening_kraj,
            MAKSIMALNO_TRENING_TRENUTAKA_NEXT,
        )
    )

    X_next_train, y_next_train = (
        napravi_dataset(
            binarna,
            next_trening_trenuci,
        )
    )

    next_model = napravi_model(
        KONFIGURACIJE_MODELA[
            izabrana_konfiguracija - 1
        ]
    )

    next_model.fit(
        X_next_train,
        y_next_train,
    )

    X_next, graf_next = (
        osobine_za_trenutak(
            binarna,
            n,
        )
    )

    next_skorovi = next_model.predict(
        X_next
    )

    print(
        "Metropolis–Hastings završna pretraga..."
    )

    (
        next_kombinacija,
        next_skor,
        prihvatanje_mh,
    ) = metropolis_hastings(
        skorovi_brojeva=next_skorovi,
        graf=graf_next,
        lambda_graf=izabrana_lambda,
        broj_lanaca=MH_NEXT_LANCI,
        broj_koraka=MH_NEXT_KORACI,
        burn_in=MH_NEXT_BURN_IN,
        seed=SEED + seed_pomeraj,
    )

    return Rezultat(
        naziv=naziv,
        broj_redova=n,
        next_kombinacija=(
            next_kombinacija
        ),
        next_skor=next_skor,
        prihvatanje_mh=(
            prihvatanje_mh
        ),
        izabrana_lambda=(
            izabrana_lambda
        ),
        izabrana_konfiguracija=(
            izabrana_konfiguracija
        ),
        validacioni_prosek=(
            validacioni_prosek
        ),
        holdout_mae=holdout_mae,
        holdout_prosek=(
            holdout_prosek
        ),
        holdout_medijana=float(
            np.median(
                holdout_pogodci
            )
        ),
        holdout_maksimum=int(
            np.max(
                holdout_pogodci
            )
        ),
        donja_granica_95=(
            donja_granica_95
        ),
        gornja_granica_95=(
            gornja_granica_95
        ),
        p_vrednost=p_vrednost,
        statisticki_pouzdano=(
            statisticki_pouzdano
        ),
        broj_validacionih_izvlacenja=len(
            validacioni_trenuci
        ),
        broj_holdout_izvlacenja=len(
            holdout_trenuci
        ),
    )


# =============================================================================
# ISPIS
# =============================================================================

def formatiraj_kombinaciju(
    kombinacija: list[int],
) -> str:
    return ", ".join(
        f"{broj:02d}"
        for broj in kombinacija
    )


def ispisi_rezultat(
    rezultat: Rezultat,
) -> None:
    print()
    print("=" * 78)
    print(rezultat.naziv)
    print("=" * 78)

    print(
        "NEXT: "
        f"{formatiraj_kombinaciju(rezultat.next_kombinacija)}"
    )

    print(
        f"CSV redova: "
        f"{rezultat.broj_redova}"
    )

    print(
        "Izabrana konfiguracija regresora: "
        f"{rezultat.izabrana_konfiguracija}"
    )

    print(
        "Izabrana grafovska lambda: "
        f"{rezultat.izabrana_lambda:.2f}"
    )

    print(
        "Validacionih izvlačenja: "
        f"{rezultat.broj_validacionih_izvlacenja}"
    )

    print(
        "Validacioni prosek pogodaka: "
        f"{rezultat.validacioni_prosek:.6f}"
    )

    print(
        "Zamrznutih holdout izvlačenja: "
        f"{rezultat.broj_holdout_izvlacenja}"
    )

    print(
        "Holdout MAE: "
        f"{rezultat.holdout_mae:.6f}"
    )

    print(
        "Holdout prosek pogodaka: "
        f"{rezultat.holdout_prosek:.6f}"
    )

    print(
        "Holdout medijana pogodaka: "
        f"{rezultat.holdout_medijana:.2f}"
    )

    print(
        "Najviše holdout pogodaka: "
        f"{rezultat.holdout_maksimum}"
    )

    print(
        "Razlika prema slučajnom očekivanju: "
        f"{rezultat.holdout_prosek - SLUCAJNO_OCEKIVANJE:+.6f}"
    )

    print(
        "Blok-bootstrap 95% interval: "
        f"[{rezultat.donja_granica_95:.6f}, "
        f"{rezultat.gornja_granica_95:.6f}]"
    )

    print(
        "Monte Karlo p-vrednost: "
        f"{rezultat.p_vrednost:.6f}"
    )

    print(
        "Završni MH skor: "
        f"{rezultat.next_skor:.6f}"
    )

    print(
        "MH stopa prihvatanja: "
        f"{rezultat.prihvatanje_mh:.2%}"
    )

    print()
    print("GLAVNI ODGOVOR")
    print("-" * 78)

    if rezultat.statisticki_pouzdano:
        print(
            "DA — na zamrznutom holdoutu utvrđena je "
            "statistički pouzdana prednost nad slučajnim "
            "očekivanjem od 1.256410 pogodaka."
        )
    else:
        print(
            "NE — na zamrznutom holdoutu nije utvrđena "
            "statistički pouzdana prednost nad slučajnim "
            "očekivanjem od 1.256410 pogodaka."
        )


# =============================================================================
# GLAVNI PROGRAM
# =============================================================================

def main() -> None:
    np.random.seed(SEED)

    print("=" * 78)
    print(
        "LOTO 7/39 — METROPOLIS–HASTINGS "
        "DISTRIBUCIJSKI SISTEM — JEDAN CSV"
    )
    print("=" * 78)

    print(f"Seed: {SEED}")

    print(
        "Teorijska stopa broja: "
        f"{TEORIJSKA_STOPA:.9f}"
    )

    print(
        "Teorijsko očekivanje pogodaka: "
        f"{SLUCAJNO_OCEKIVANJE:.9f}"
    )

    print(
        "Ukupno mogućih kombinacija: "
        f"{UKUPNO_KOMBINACIJA:,}"
    )

    rezultat = obradi_igru(
        naziv="Loto",
        putanja=ZAJEDNICKI_CSV,
        seed_pomeraj=0,
    )

    print()
    print()
    print("#" * 78)
    print("KONAČNA NEXT PREDIKCIJA")
    print("#" * 78)

    ispisi_rezultat(rezultat)


if __name__ == "__main__":
    main()



"""
==============================================================================
LOTO 7/39 — METROPOLIS–HASTINGS DISTRIBUCIJSKI SISTEM — JEDAN CSV
==============================================================================
Seed: 39
Teorijska stopa broja: 0.179487179
Teorijsko očekivanje pogodaka: 1.256410256
Ukupno mogućih kombinacija: 15,380,937

==============================================================================
Obrada: Loto
==============================================================================
CSV: /data/loto7_4680_k71.csv
Broj redova: 4680
Prvi red se tretira kao najstariji.
Poslednji red se tretira kao najnoviji.
Hronološki izbor regresora i grafovske lambda vrednosti...
  Model 1, lambda=0.00: prosek=1.283333, MAE=0.294435
  Model 1, lambda=0.10: prosek=1.208333, MAE=0.294435
  Model 1, lambda=0.25: prosek=1.258333, MAE=0.294435
  Model 1, lambda=0.50: prosek=1.250000, MAE=0.294435
  Model 1, lambda=1.00: prosek=1.325000, MAE=0.294435
  Model 2, lambda=0.00: prosek=1.166667, MAE=0.294909
  Model 2, lambda=0.10: prosek=1.133333, MAE=0.294909
  Model 2, lambda=0.25: prosek=1.100000, MAE=0.294909
  Model 2, lambda=0.50: prosek=1.300000, MAE=0.294909
  Model 2, lambda=1.00: prosek=1.275000, MAE=0.294909
  Model 3, lambda=0.00: prosek=1.075000, MAE=0.294749
  Model 3, lambda=0.10: prosek=1.141667, MAE=0.294749
  Model 3, lambda=0.25: prosek=1.183333, MAE=0.294749
  Model 3, lambda=0.50: prosek=1.175000, MAE=0.294749
  Model 3, lambda=1.00: prosek=1.125000, MAE=0.294749
Izabrana konfiguracija regresora: 1
Izabrana grafovska lambda: 1.00
Validacioni prosek pogodaka: 1.325000
Zamrznuta holdout provera...
  Holdout 1/120
  Holdout 20/120
  Holdout 40/120
  Holdout 60/120
  Holdout 80/120
  Holdout 100/120
  Holdout 120/120
Završna obuka nad svim dostupnim poznatim metama...
Metropolis–Hastings završna pretraga...


##############################################################################
KONAČNA NEXT PREDIKCIJA
##############################################################################

==============================================================================
Loto
==============================================================================
NEXT: 09, x, 13, y, 24, z, 34
CSV redova: 4680
Izabrana konfiguracija regresora: 1
Izabrana grafovska lambda: 1.00
Validacionih izvlačenja: 120
Validacioni prosek pogodaka: 1.325000
Zamrznutih holdout izvlačenja: 120
Holdout MAE: 0.295844
Holdout prosek pogodaka: 1.025000
Holdout medijana pogodaka: 1.00
Najviše holdout pogodaka: 4
Razlika prema slučajnom očekivanju: -0.231410
Blok-bootstrap 95% interval: [0.850000, 1.175000]
Monte Karlo p-vrednost: 0.997700
Završni MH skor: 25.159451
MH stopa prihvatanja: 2.65%

GLAVNI ODGOVOR
------------------------------------------------------------------------------
NE — na zamrznutom holdoutu nije utvrđena statistički pouzdana prednost nad slučajnim očekivanjem od 1.256410 pogodaka.
"""



"""
Metropolis-Hastings može da se primeni na Loto 7/39, 
ali kao algoritam za uzorkovanje kombinacija iz zadate raspodele 
— nije samostalni prediktivni model.

Postoji 15.380.937 kombinacija, pa se može izračunati skor svake kombinacije. 
Zato Metropolis-Hastings ovde nije nužno bolji od potpunog pregleda; 
koristan je kada je izračunavanje skora kombinacije skupo 
ili kada želiš uzorkovanje iz cele distribucije umesto samo kombinacije sa najvećim skorom.

Za jedan CSV-a Metropolis–Hastings:
1. Iz Loto CSV-a model izračuna kontinuirane skorove 39 brojeva i grafovske skorove parova.
2. Metropolis–Hastings pretražuje prostor od 15.380.937 kombinacija koristeći samo te Loto skorove.
3. Kombinacija sa najvećim pronađenim ukupnim skorom postaje Loto NEXT.
 

Metropolis–Hastings je završni pretraživač; regresor i graf određuju ciljnu raspodelu. 

Distribucijski gradient-boosting regresor + grafovski skor parova + Metropolis–Hastings + hronološka validacija.

Poboljšanje može se utvrditi samo rezultatima holdout testa. 
"""
