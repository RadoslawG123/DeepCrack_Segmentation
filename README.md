# Segmentacja pęknięć betonu i asfaltu — DeepCrack

Projekt z przedmiotu *Przetwarzanie i analiza cyfrowych danych przestrzennych i budowlanych metodami sztucznej inteligencji*.

**Autorzy:** Laura Wojtyla, Katarzyna Drożdż, Oliwia Grądzka, Natalia Kadłuczka, Radosław Górecki

Porównanie trzech architektur segmentacji semantycznej (**U-Net**, **DeepLabV3+**, **FPN**) w dwóch konfiguracjach: z enkoderem pretrenowanym na ImageNet oraz trenowanym od zera — łącznie 6 treningów.

- Prezentacja: [`Segmentacja pęknięć.pptx`](Segmentacja%20p%C4%99kni%C4%99%C4%87.pptx)
- Kod: [`deepcrack_segmentacja.ipynb`](deepcrack_segmentacja.ipynb) (Google Colab, GPU T4)

---

## 1. Problem i zbiór danych

| | |
|---|---|
| **Typ uszkodzenia** | pęknięcia i rysy (szerokość od 1 do 180 px) |
| **Obiekty** | nawierzchnie asfaltowe, ściany i elementy betonowe |
| **Typ problemu** | segmentacja semantyczna binarna: rysa / tło |
| **Zbiór** | DeepCrack — 537 zdjęć 544×384 px z maskami pikselowymi |
| **Podział** | 240 train / 60 val / 237 test (walidacja wylosowana z oficjalnego zbioru treningowego, seed 42) |
| **Niezbalansowanie klas** | ok. 3,5% pikseli to rysa |

**Źródło:** Liu Y., Yao L., Lu X., Xie R., Li L., *DeepCrack: A deep hierarchical feature learning architecture for crack segmentation*, Neurocomputing (2019). Repozytorium: https://github.com/yhlleo/DeepCrack

**Licencja zbioru:** według autorów użycie jest ograniczone do niekomercyjnych badań i celów edukacyjnych. Zdjęcia nie są kopiowane do tego repozytorium — notatnik pobiera je automatycznie z repozytorium autorów. Plik [`data/split.csv`](data/split.csv) zawiera użyty podział na zbiory treningowy, walidacyjny i testowy.

## 2. Plan eksperymentu

Wszystkie trzy architektury mają ten sam enkoder **ResNet34** (implementacja: `segmentation_models_pytorch`), więc porównywane są dekodery.

| Parametr | Konfiguracja 1 | Konfiguracja 2 |
|---|---|---|
| **Wagi enkodera** | **ImageNet** | **od zera** |
| Seed | 42 | 42 |
| Podział danych | 240 / 60 / 237 | ten sam |
| Wejście w treningu | losowe wycinki 384×384 | to samo |
| Ewaluacja | pełny obraz 544×384 | ta sama |
| Augmentacja | odbicia, obrót o 90°, jasność/kontrast | ta sama |
| Epoki / batch | 40 / 8 | 40 / 8 |
| Optymalizator | AdamW, lr 3e-4, weight decay 1e-4, cosine annealing | ten sam |
| Funkcja straty | Dice + BCE | ta sama |
| Wybór checkpointu | najlepsze val IoU | ten sam |

### Hipotezy
- **H1:** pretrening da wyższe IoU dla wszystkich trzech architektur (o 5–15 pp), ponieważ zbiór treningowy jest mały.
- **H2:** największy spadek bez pretreningu wystąpi dla DeepLabV3+.
- **H3:** w konfiguracji 1 U-Net i FPN osiągną wyższe IoU niż DeepLabV3+ (cienkie rysy).
- **H4:** trening od zera zbiega wolniej, a jego błędy to głównie przeoczone rysy (niższy recall).

### Budżet obliczeniowy
Pomiar jednej epoki na 100 obrazach przed treningiem: szacunek **33 min** na cały plan. Rzeczywisty czas 6 treningów: **ok. 25 min** (Google Colab, GPU T4).

## 3. Wyniki (zbiór testowy, 237 obrazów)

Metryki dla klasy „rysa”, liczone globalnie po wszystkich pikselach zbioru testowego, próg 0,5.

| Model | Konfiguracja | IoU | F1 | Precision | Recall | Val IoU | Trening [min] | Inferencja [ms] |
|---|---|---|---|---|---|---|---|---|
| U-Net | pretrening | 0,723 | 0,840 | 0,913 | 0,777 | 0,805 | 4,7 | 16,6 |
| DeepLabV3+ | pretrening | 0,709 | 0,830 | 0,881 | 0,784 | 0,759 | 3,8 | 15,8 |
| FPN | pretrening | 0,715 | 0,834 | 0,888 | 0,785 | 0,771 | 4,0 | 13,9 |
| U-Net | od zera | **0,726** | **0,841** | 0,902 | 0,789 | 0,756 | 4,6 | 17,3 |
| DeepLabV3+ | od zera | 0,669 | 0,802 | 0,872 | 0,742 | 0,707 | 3,9 | 16,4 |
| FPN | od zera | 0,688 | 0,815 | 0,875 | 0,763 | 0,733 | 3,9 | 14,5 |

**Efekt rezygnacji z pretreningu** (konfiguracja 2 − konfiguracja 1):

| Model | Δ IoU | Δ F1 |
|---|---|---|
| DeepLabV3+ | −3,97 pp | −2,78 pp |
| FPN | −2,69 pp | −1,86 pp |
| U-Net | +0,28 pp | +0,19 pp |

![IoU na zbiorze testowym](results/test_iou_bar.png)
![Krzywe uczenia](results/learning_curves.png)

## 4. Wnioski

| Hipoteza | Wynik | Werdykt |
|---|---|---|
| H1 | DeepLabV3+ −4,0 pp, FPN −2,7 pp bez pretreningu; U-Net +0,3 pp | częściowo potwierdzona |
| H2 | największy spadek dla DeepLabV3+ | potwierdzona |
| H3 | 0,723 > 0,715 > 0,709 — rozrzut 1,4 pp przy jednym seedzie | kierunek zgodny, różnica w granicach szumu |
| H4 | wolniejsza zbieżność: tak; niższy recall dla DeepLabV3+ i FPN, dla U-Neta nie | częściowo potwierdzona |

- **Najskuteczniejszy model: U-Net** (IoU 0,72–0,73), jako jedyny niezależny od pretreningu — prawdopodobnie dzięki połączeniom skrótowym, które przekazują cechy wysokiej rozdzielczości bezpośrednio do dekodera.
- **Efekt pretreningu zależy od architektury** i nie jest uniwersalny.
- We wszystkich treningach precision > recall: modele częściej przeoczają rysy, niż wykrywają fałszywe.
- **Najgorsze przypadki** to trzy zdjęcia serii IMG24 (gładki tynk, niski kontrast), które występują wyłącznie w zbiorze testowym: z pretreningiem wszystkie modele mają na nich IoU ≤ 0,07; bez pretreningu jedynie FPN wykrywa fragmenty rys (IoU do 0,28).

![Najgorsze przypadki](results/pred_worst.png)

### Ograniczenia
- jeden seed na trening — różnic poniżej 1–2 pp nie da się rozstrzygnąć,
- zbiór walidacyjny (60 obrazów) różni się rozkładem od testowego (val IoU 0,71–0,81, test IoU 0,67–0,73),
- IoU jest bardzo surowe dla rys o szerokości 2–3 px; próg 0,5 nie był dobierany.

### Dalsze kroki
Kilka seedów na trening (średnia ± odchylenie), F1 z tolerancją 2–3 px, podział danych uwzględniający serie zdjęć, dobór progu na zbiorze walidacyjnym, test generalizacji na innym zbiorze (np. dacl10k).

## 5. Uruchomienie

1. Otworzyć `deepcrack_segmentacja.ipynb` w Google Colab z GPU T4.
2. Uruchomić komórki po kolei — notatnik instaluje biblioteki i pobiera zbiór DeepCrack.
3. Wyniki (tabele CSV, wykresy, wagi modeli) zapisują się na Google Drive w `BudownictwoAI/deepcrack_runs`.

Lokalnie: `pip install -r requirements.txt` (wymagane GPU z CUDA, `USE_DRIVE = False`).

## 6. Struktura repozytorium

```
├── README.md
├── requirements.txt
├── deepcrack_segmentacja.ipynb     # kod: dane, modele, trening, ewaluacja, wykresy
├── Segmentacja pęknięć.pptx        # prezentacja
├── data/
│   └── split.csv                   # podział train/val/test (ścieżki do plików DeepCrack)
└── results/
    ├── timing_estimate.csv         # pomiar czasu przed treningiem
    ├── test_results.csv            # wyniki 6 treningów
    ├── config_effect.csv           # Δ IoU / Δ F1
    ├── dataset_samples.png         # przykładowe obrazy z maskami
    ├── learning_curves.png         # krzywe uczenia
    ├── test_iou_bar.png            # IoU na zbiorze testowym
    ├── test_per_image_iou.csv      # IoU każdego obrazu testowego dla 6 modeli
    ├── per_image_iou_box.png       # rozkład IoU dla pojedynczych obrazów
    ├── pred_*.png                  # przykładowe predykcje
    └── <model>__<konfiguracja>/    # 6 folderów, po jednym na trening
        ├── history.csv             # loss, val IoU, val F1, lr i czas każdej epoki
        └── meta.json               # konfiguracja, najlepsze val IoU, czas treningu, liczba parametrów
```

Wagi modeli (`best.pth`, ok. 90 MB każdy) nie są przechowywane w repozytorium ze względu na limit rozmiaru plików GitHuba.

Foldery `<model>__<konfiguracja>`: `UNet`, `DeepLabV3Plus`, `FPN` × `C1_pretrained` (enkoder pretrenowany na ImageNet), `C2_scratch` (enkoder trenowany od zera).
