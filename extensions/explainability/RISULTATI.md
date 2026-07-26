# Attribuzione esatta in DGCDR — pipeline, risultati, conclusioni sul Contributo 1

Documento di lavoro, aggiornato al 26 luglio 2026. Branch `explainability-1`.
Cinque checkpoint analizzati: CDs→Instruments, Elec→Cloth, e Cloth→Elec a tre
valori di `cl_org_weight`.

---

## 1. Cosa fa la pipeline

DGCDR divide i gusti di ogni utente in due parti:

- **shared** — i gusti che l'utente porta dall'altro dominio;
- **specific** — i gusti che valgono solo nel dominio in cui stiamo raccomandando.

Poi le rimette insieme e calcola un punteggio per ogni prodotto.

La pipeline fa una cosa sola: **riapre quel punteggio e dice quanta parte viene da ciascuna delle due**.

Il risultato è **τ (transfer ratio)**, un numero tra 0 e 1, calcolato per ogni singola raccomandazione:

- **τ ≈ 1** → "ti consiglio questo perché ho capito i tuoi gusti nell'altro dominio"
- **τ ≈ 0** → "ti consiglio questo per come ti sei comportato qui"
- **τ ≈ 0.5** → le due cose pesano uguale

### Perché non è il solito explainer

Gli explainer basati su LLM passano all'LLM la cronologia dell'utente e chiedono una motivazione plausibile. Il testo suona bene, ma **niente garantisce che c'entri con quello che il modello ha calcolato**.

Qui il meccanismo è aritmetico. DGCDR somma le sue componenti e il punteggio è un prodotto scalare, quindi si spacca in modo esatto:

```
punteggio = (contributo shared) + (contributo specific) + (contributo collaborativo grezzo)
```

Non è una stima e non è un surrogato come LIME o SHAP: è la stessa aritmetica del modello, riscritta. Ogni esecuzione ricompone i pezzi e li confronta con l'output vero; se non tornano, si ferma. Sui cinque checkpoint l'errore relativo sta tra 1e-07 e 3e-07.

L'LLM arriva **dopo**: riceve i numeri e li traduce in linguaggio naturale, con il divieto di contraddirli.

### I pezzi

| file | cosa fa |
|---|---|
| `channels.py` | riapre gli embedding nelle componenti e verifica che tornino |
| `attribution.py` | matrice dei contributi, τ, rilevanza delle raccomandazioni |
| `metadata.py` | traduce gli ID in titoli e categorie leggibili (opzionale) |
| `verbalize.py` | prompt vincolato e chiamata all'LLM (opzionale) |
| `explain_dgcdr.py` | lo script da lanciare |
| `test_decomposition.py` | autotest: 8 controlli, da rilanciare su ogni nuovo modello |

La pipeline **non tocca il codice del modello**: usa solo le funzioni pubbliche di DGCDR.

### Quando si rifiuta di funzionare

Meglio un errore che una spiegazione falsa. Lo script si blocca con fusione non additiva (`fuse_mode='concat'`), disentanglement disattivato, o modello non in modalità valutazione.

---

## 2. Gli esperimenti

| | CDs→Instruments | Elec→Cloth | Cloth→Elec (×3) |
|---|---|---|---|
| utenti | 1.842 | 35.827 | 35.827 |
| prodotti (target) | 3.610 | 72.669 | 62.548 |
| interazioni (target) | ~16 mila | ~847 mila | ~812 mila |
| iperparametri | default del repo | paper sez. 2.1 | paper sez. 2.2 |
| `cl_org_weight` | 1 | 0.1 | **0.01 / 1 / 10** |

Il primo serviva a validare il codice. Gli altri sono gli esperimenti veri.

---

## 3. Validazione: riproduciamo il paper

Prima di qualunque conclusione va stabilito che stiamo misurando il DGCDR vero e non un training andato male. Confronto con la Tabella 3 del paper:

| | | paper | nostro run | scarto |
|---|---|---|---|---|
| **Elec→Cloth** (target Cloth) | Recall | .0260 | .0253 | −2.7% |
| | MRR | .0278 | .0281 | +1.1% |
| | HR | .0876 | .0869 | −0.8% |
| | NDCG | .0173 | .0173 | **0%** |
| **Cloth→Elec** (target Elec) | Recall | .0403 | .0397 | −1.5% |
| | MRR | .0353 | .0349 | −1.1% |
| | HR | .1304 | .1276 | −2.1% |
| | NDCG | .0247 | .0244 | −1.2% |

Scarti tra l'1% e il 3%, cioè normale variabilità da seed. **Tutto quello che segue vale quindi per il modello pubblicato**, non per una nostra versione difettosa. È il punto che rende difendibile il resto del documento.

---

## 4. I risultati

### 4.0 Lo strumento di misura era difettoso

`evaluate_disentanglement.py` aveva tre difetti, e **tutti e tre gonfiavano il disentanglement apparente**. La matematica di dCor era corretta — verificata su casi noti, restituisce esattamente 1 per input identici, riscalati e ruotati. Il problema stava attorno.

1. **Probe su feature non standardizzate.** I valori stanno intorno a 0.02, quindi la penalità L2 di default dominava e i classificatori andavano in underfitting. Nessun warning. Su Elec→Cloth il probe lineare legge 98.82% con lo scaler e 81.79% senza: **diciassette punti di sottostima**.
2. **dCor confrontato con uno zero irraggiungibile.** È la forma V-statistic, distorta verso l'alto a N finito e alta dimensione: due gaussiane *indipendenti* con N=5000 e D=256 danno 0.41. Ora lo script permuta le righe di uno spazio, ottiene il null empirico e riporta lo scarto.
3. **19 GB di memoria** per la matrice delle distanze a 35k utenti. Ora campiona: a 5.000 la stima cambia di un centesimo.

**Questo è già un risultato.** Con il difetto 1, Elec→Cloth sembrava avere un disentanglement score del 18%. Il valore vero è 1.18%. Un probe sotto-specificato sovrastima il disentanglement in modo silenzioso, e la trappola vale per chiunque riporti queste metriche.

### 4.1 L'ortogonalità non compra indipendenza

È il risultato principale, e lo sweep controllato lo chiude.

Il modello è addestrato a rendere i due canali **perpendicolari** — il coseno misura quello. Ma l'obiettivo è che il canale *shared* abbia dimenticato da quale dominio viene, e quello si misura altrimenti: con la distance correlation, e chiedendo a un classificatore di indovinare il dominio guardando solo quel canale.

Sweep su Cloth→Elec, **una sola variabile che cambia**, tutto il resto fisso:

| `cl_org_weight` | 0.01 | 1 | 10 |
|---|---|---|---|
| **Coseno** (target) — *ideale 0* | 0.2164 | 0.0119 | **0.0068** |
| dCor gap (target) — *ideale 0* | +0.7305 | +0.6726 | **+0.6479** |
| dCor gap (source) — *ideale 0* | +0.8032 | +0.7631 | **+0.7566** |
| Probe lineare e^c — *ideale 50%* | 99.19% | 98.76% | **99.12%** |
| Probe MLP e^c — *ideale 50%* | 99.92% | 99.77% | **99.91%** |
| Recall@20 | 0.0395 | 0.0397 | **0.0401** |

**Aumentando il peso di mille volte, i canali diventano trentadue volte più perpendicolari, il gap di dipendenza migliora dell'11%, e la fuga di informazione sul dominio resta esattamente dov'era: 99%.**

Il coseno crolla in modo monotono: la loss fa il mestiere per cui è scritta. I probe non si muovono affatto — 99.19 → 98.76 → 99.12 è rumore. Il gap di dCor migliora in modo monotono ma minuscolo.

**Il calcolo che chiude la questione.** Il gap scende in modo circa lineare nel logaritmo del peso: −0.029 per decade tra 0.01 e 1, −0.025 nell'ultima decade. A questo ritmo, per portarlo da 0.648 a zero servirebbero **oltre venti decadi**, cioè un `cl_org_weight` dell'ordine di 10²³. Non è che la leva sia inefficace: è inefficace di un fattore astronomico. Questo è più forte di "non migliora", perché è quantificato e non si può obiettare "avresti dovuto spingere di più".

**Non c'è un trade-off da invocare.** Recall@20 va 0.0395 → 0.0397 → 0.0401: sale impercettibilmente. Anche con peso 10, quando il termine di ortogonalità domina la loss, l'accuratezza non se ne accorge. Il vincolo è semplicemente **disaccoppiato** da ciò che fa funzionare il modello.

**Un effetto funzionale c'è, ma è un altro.** τ si allarga in modo monotono (deviazione 0.065 → 0.080 → 0.091) e le raccomandazioni native-driven triplicano (4.8% → 15.3%). L'ortogonalità sposta il baricentro verso il canale specific. Fa qualcosa, non quello che dovrebbe.

**L'analogia.** Prendi dei punti su una circonferenza. Le coordinate x e y hanno correlazione zero: sono "perpendicolari" in senso statistico. Eppure conoscendo x sai quanto vale y, a meno del segno. Sono scorrelate ma dipendenti. Il modello azzera il prodotto scalare e ottiene precisamente quello.

**Contraddice una scelta di design dichiarata.** Il paper scrive di aver provato la cosine similarity, di aver ottenuto risultati peggiori, e di aver scelto l'ortogonalità perché *"provides a clearer separation between subspaces, resulting in more robust representation disentanglement"*. La separazione è chiara nello spazio, non nell'informazione.

### 4.2 L'attention è quasi inerte, e la Figura 3 non è riproducibile

Il paper dedica una figura alla distribuzione dell'attention tra i due canali, e ci costruisce sopra due affermazioni interpretative: domini poco correlati peserebbero di più le shared, domini di contenuto (Movie) le specific.

Eq. (4) del paper è esattamente ciò che il codice implementa e ciò che abbiamo misurato:

```
[a_c, a_s] = softmax( e_g · [e_c, e_s] / √d )
```

| Elec&Cloth | Figura 3 | nostra misura |
|---|---|---|
| shared | 80.55% / 80.78% | **50.10% / 49.52%** |
| specific | 19.45% / 19.22% | 49.90% / 50.48% |

Per produrre 80.55/19.45 servirebbe un gap di logit pari a **22.74**. Nel nostro modello è **+0.06** sul source e **−0.31** sul target, con deviazione ~0.7–0.9.

Abbiamo escluso che la figura riporti una quantità diversa: né i pesi di attention (50.10%), né la proporzione delle norme dopo l'attention (49.52%), né quella prima (49.43%) si avvicinano a 80/20.

Il motivo dell'inerzia è architetturale: i logit vengono divisi per √d, e con d = 256 la scala è **16**. Differenze già piccole vengono schiacciate, e il softmax restituisce quasi sempre mezzo e mezzo.

**Conseguenza per l'interpretazione di τ.** Scomponendo la varianza dei log-odds di τ, l'attention spiega l'**1.3%** e la geometria degli item il **92.5%**. Forzando l'attention a 0.5/0.5 esatti, τ passa da 0.4159 a 0.4212 con deviazione quasi invariata. Quindi **τ varia perché i prodotti si allineano diversamente con i due sottospazi, non perché il modello decida diversamente da utente a utente.**

**Limite.** Abbiamo misurato solo la coppia Elec/Cloth. La Figura 3 ha sei barre su tre coppie, e la più estrema è Douban Movie a 28.04/71.96. Per ora si può dire "per Elec/Cloth non è riproducibile", non "la Figura 3 è sbagliata". Il test decisivo è allenare Douban Movie↔Book.

### 4.3 τ come strumento diagnostico

Sul dataset piccolo τ era una costante: 0.4972 con deviazione **0.006**, uguale per ogni utente e ogni prodotto. Sembrava un bug. Non lo era: lì i due canali erano **lo stesso vettore** (coseno 0.997), quindi contribuivano per forza allo stesso modo.

Il meccanismo del collasso: entrambi i canali si ottengono filtrando lo stesso vettore con due "rubinetti" appresi, e i rubinetti avevano imparato valori quasi identici, 0.408 contro 0.409. Da lì degenera tutto, compresa l'attention, a 0.4999 contro 0.5001.

| | CDs→Instr. | Elec→Cloth | Cloth→Elec (org 1) |
|---|---|---|---|
| τ pooled | 0.4967 | 0.4245 | 0.5031 |
| deviazione | **0.006** | 0.084 | 0.080 |
| τ per utente (min–max) | ~0.50 piatto | 0.242 – 0.741 | 0.310 – 0.733 |

**Due situazioni opposte dietro lo stesso numero.** τ vale ~0.50 sia su CDs→Instruments sia su Cloth→Elec. Nel primo caso è collasso (deviazione 0.006, tutti identici); nel secondo una distribuzione vera centrata a metà. **Chi guarda solo la media conclude il contrario del vero: il segnale è la dispersione.**

È il contributo che τ ha effettivamente dato: una spia di collasso leggibile a colpo d'occhio, definita sulla singola raccomandazione invece che sull'intero spazio latente.

### 4.4 Il risultato sul transfer agli utenti abituali: **ritirato**

Nelle prime due misure sembrava emergere che più un utente ha storia nel dominio target, più la raccomandazione dipende dal transfer — il contrario dell'assunzione comune sul cross-domain. Con quattro misure il quadro si scioglie:

| run | τ (storia 5–19) | τ (storia 20+) | differenza |
|---|---|---|---|
| Elec→Cloth, org 0.1 | 0.4145 | 0.4813 | **+0.067** |
| Cloth→Elec, org 1 | 0.5010 | 0.5231 | **+0.022** |
| Cloth→Elec, org 0.01 | 0.5000 | 0.4866 | **−0.013** |
| Cloth→Elec, org 10 | 0.4758 | 0.4592 | **−0.017** |

L'effetto **cambia segno** al variare di un solo peso di loss, sullo stesso dataset, nella stessa direzione, con gli stessi utenti. Anche la correlazione crolla da +0.41 a +0.11 tra le due direzioni.

Non è un effetto debole: è un artefatto degli iperparametri. Va ritirato, non ridimensionato. Resta utile come informazione negativa — dice che τ, aggregata per fasce di utenti, **non è una quantità stabile** nel modello attuale, il che è coerente con un canale "shared" che non rappresenta stabilmente ciò che il suo nome promette.

---

## 5. Conclusioni sul Contributo 1

**L'attribuzione funziona come strumento.** È esatta per costruzione, si autoverifica a ogni esecuzione, si rifiuta di produrre spiegazioni quando non può garantirle, e regge su cinque checkpoint e quattro configurazioni. Come contributo tecnico è solido e difendibile.

**Ma la sua resa scientifica diretta è diagnostica, non esplicativa.** Vale la pena essere onesti su cosa τ ha e non ha prodotto:

| cosa speravamo | cosa è successo |
|---|---|
| τ rivela chi beneficia del transfer | ritirato, artefatto degli iperparametri (§4.4) |
| τ come misura di influenza cross-domain | l'etichetta "shared" non regge: canale al 99% riconoscibile per dominio |
| τ come spia di collasso | ✅ funziona, ed è il contributo che resta |
| l'attention rivela le preferenze di fusione | l'attention spiega l'1.3% di τ, è quasi inerte |

**I risultati che sopravvivono non vengono da τ.** Vengono dall'infrastruttura di misura costruita attorno — probe corretti, gap di dCor, sweep controllato — e dal confronto con quanto il paper dichiara. Il Contributo 1 è servito soprattutto a rendere possibili quelle misure, e a fornire una diagnosi (§4.3) che le metriche esistenti non danno.

**La spina dorsale del lavoro si è spostata.** Non è più "spieghiamo le raccomandazioni cross-domain", ma:

> **I meccanismi di DGCDR non fanno quello che il paper dichiara facciano**, dimostrato con tre reperti indipendenti su un modello che riproduce i risultati pubblicati entro il 3%:
> 1. l'ortogonalità non compra indipendenza, e non lo farebbe nemmeno con pesi 10²³ volte maggiori;
> 2. l'attention è quasi inerte, e la figura che la descrive non è riproducibile;
> 3. il canale "domain-shared" resta riconoscibile per dominio al 99%.

È un paper critico verso un lavoro esistente. È una scelta che va fatta consapevolmente, ma i tre reperti sono più solidi di qualunque risultato positivo che avessimo in mano all'inizio — e la pipeline di attribuzione resta il contributo metodologico che li rende misurabili.

---

## 6. Limiti

- **Un seed per configurazione.** Niente è stato replicato con seed diversi.
- I confronti tra le due *direzioni* cambiano più variabili insieme. Lo sweep interno a Cloth→Elec invece è a variabile singola ed è quello su cui poggia §4.1.
- I dataset sono 10-core: **nessun utente davvero freddo**.
- τ descrive **come è composto** il punteggio, non cosa succederebbe togliendo un canale e ricalcolando la classifica.
- τ alto **non** significa raccomandazione migliore. È un'attribuzione, non una valutazione.
- La scomposizione esatta richiede `fuse_mode='attention'`, e solo gli utenti sovrapposti hanno un τ definito.
- I probe sono **a soffitto** (98–100%): lì non distinguono più configurazioni diverse, e il confronto va appoggiato sul gap di dCor.
- Il gap di dCor dipende dal numero di utenti campionati: confrontabile solo a `--dcor_sample` uguale. CDs→Instruments (1.842 utenti sovrapposti) resta fuori confronto.
- **Bug noto in `dgcdr.py`**, rilevante per il Contributo 2: il caricamento degli embedding testuali usa l'ID fuso per indicizzare l'array del dominio source, che è compattato. Su CDs/Instruments questo mappa correttamente solo i 3.609 item del target, ne mappa 23 **sul prodotto sbagliato**, e lascia a zero tutti i 3.609 del source — con la semantic loss che li tira comunque verso una costante. Nessuno dei run qui usava `use_text_embeddings=True`.
- **Previsione sbagliata**, annotata: mi aspettavo τ più basso su Cloth→Elec. È uscito più alto (0.503 contro 0.425).
- **Affermazione ritirata**: avevo scritto che ogni misura di indipendenza peggiora aumentando `cl_org_weight`, basandomi sul confronto confondato tra direzioni. Nello sweep pulito il gap di dCor migliora leggermente e i probe restano fermi.

---

## 7. Prossimi passi

**1. Douban Movie↔Book.** Il run con il miglior rapporto valore/costo: testa il punto più estremo della Figura 3 (28.04/71.96) e aggiunge una terza coppia di domini a tutto il resto. Se anche lì l'attention esce ~50/50, §4.2 è chiuso su tutte e tre le coppie del paper.

**2. Loss domain-adversariale** (branch `domain-adversarial`, già implementata). È l'unico intervento che attacca il probe al 99%, l'unica misura che lo sweep ha lasciato immobile. Da valutare sul gap di dCor e su Recall, non sul probe — che stiamo ottimizzando.

**3. Replica con più seed**, per stabilire quanto sono stabili τ e le misure di disentanglement.

**4. Controfattuale**: azzerare il canale shared, ricalcolare le classifiche, passare dall'attribuzione alla causalità.

**5. Contributo 2 (concept naming).** Richiede prima la correzione del bug sulla mappatura degli ID nei text embedding.
