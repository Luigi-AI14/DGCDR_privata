# Attribuzione esatta in DGCDR — cosa fa la pipeline e cosa abbiamo trovato

Documento di lavoro, aggiornato al 25 luglio 2026. Branch `explainability-1`.
Tre run analizzati: CDs→Instruments, Elec→Cloth, Cloth→Elec.

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

Gli explainer basati su LLM funzionano così: passano all'LLM la cronologia dell'utente e gli chiedono una motivazione plausibile. Il testo suona bene. Ma **niente garantisce che c'entri con quello che il modello ha calcolato**: è una giustificazione costruita a posteriori.

Qui il meccanismo è aritmetico. DGCDR somma le sue componenti, e il punteggio è un prodotto scalare. Quindi il punteggio si spacca in modo esatto:

```
punteggio = (contributo shared) + (contributo specific) + (contributo collaborativo grezzo)
```

Non è una stima e non è un surrogato come LIME o SHAP. È la stessa aritmetica del modello, riscritta.

Ogni esecuzione ricompone i pezzi e li confronta con l'output vero del modello. Se non tornano, si ferma. Sui tre run l'errore relativo sta tra 1e-07 e 3e-07: solo arrotondamento in virgola mobile.

L'LLM arriva **dopo**, e fa molto meno di quanto si creda: riceve i numeri e li traduce in linguaggio naturale, con il divieto esplicito di contraddirli. Non decide lui il perché.

### I pezzi

| file | cosa fa |
|---|---|
| `channels.py` | riapre gli embedding nelle componenti e verifica che tornino |
| `attribution.py` | calcola la matrice dei contributi e τ |
| `metadata.py` | traduce gli ID dei prodotti in titoli leggibili (opzionale) |
| `verbalize.py` | costruisce il prompt vincolato e interroga l'LLM (opzionale) |
| `explain_dgcdr.py` | lo script da lanciare |
| `test_decomposition.py` | autotest: 8 controlli, da rilanciare su ogni nuovo modello |

La pipeline **non tocca il codice del modello**: usa solo le funzioni pubbliche di DGCDR. Il training resta identico.

### Quando si rifiuta di funzionare

Meglio un errore che una spiegazione falsa. Lo script si blocca in tre casi: fusione non additiva (`fuse_mode='concat'`), disentanglement disattivato, modello non in modalità valutazione. In tutti e tre la scomposizione non sarebbe esatta.

---

## 2. I tre esperimenti

| | CDs → Instruments | Elec → Cloth | Cloth → Elec |
|---|---|---|---|
| utenti | 1.842 | 35.827 | 35.827 |
| prodotti (target) | 3.610 | 72.669 | 62.548 |
| interazioni (target) | ~16 mila | ~847 mila | ~812 mila |
| iperparametri | default del repo | paper sez. 2.1 | paper sez. 2.2 |
| `cl_org_weight` | 1 | 0.1 | 1 |
| `cl_sim_weight` | 0.01 | 0.1 | 0.01 |
| `item_cl_weight` | 0.01 | 0.1 | 0.01 |
| Recall@20 (test) | 0.0648 | 0.0253 | 0.0397 |

Il primo serviva a validare il codice. Gli altri due sono gli esperimenti veri, sulle due direzioni della stessa coppia di domini.

---

## 3. I risultati

### 3.0 Lo strumento di misura era difettoso

`evaluate_disentanglement.py` misura tutto quello che segue, e aveva tre difetti. La matematica di dCor era corretta — verificata su casi noti, restituisce esattamente 1 per input identici, riscalati e ruotati. Il problema stava attorno. **Tutti e tre i difetti gonfiavano il disentanglement apparente.**

1. **Probe su feature non standardizzate.** I valori stanno intorno a 0.02, quindi la penalità L2 di default dominava e i classificatori andavano in underfitting. Nessun warning: semplicemente non usavano il segnale disponibile. Su Elec→Cloth il probe lineare legge 98.82% con lo scaler e 81.79% senza. **Diciassette punti di sottostima.**
2. **dCor confrontato con uno zero irraggiungibile.** È la forma V-statistic, distorta verso l'alto a N finito e alta dimensione: due gaussiane *indipendenti* con N=5000 e D=256 danno 0.41. Il valore grezzo non significa nulla da solo. Ora lo script permuta le righe di uno dei due spazi, ottiene il null empirico e riporta lo scarto.
3. **19 GB di memoria** per la matrice delle distanze a 35k utenti. Ora campiona: a 5.000 utenti la stima cambia di un centesimo.

**Questo è già un risultato.** Con il difetto 1, Elec→Cloth sembrava avere un disentanglement score del 18%: mediocre ma non nullo. Il valore vero è 1.18%. Un probe sotto-specificato sovrastima il disentanglement in modo silenzioso, e la stessa trappola vale per chiunque riporti queste metriche.

### Risultato 1 — L'ortogonalità non compra indipendenza

È il risultato principale.

Il modello è addestrato a rendere i due canali **perpendicolari**, e il coseno misura proprio quello. Ma l'obiettivo non è la perpendicolarità: è che il canale *shared* abbia dimenticato da quale dominio viene. Quello si misura in altro modo — con la distance correlation, e chiedendo a un classificatore di indovinare il dominio guardando solo quel canale.

I numeri sono quelli **ricalcolati** dopo la correzione descritta in §3.0.

| | CDs→Instruments | Elec→Cloth | **Cloth→Elec** |
|---|---|---|---|
| `cl_org_weight` | 1 | 0.1 | **1** |
| Coseno (target) — *ideale 0* | 0.9973 | 0.0365 | **0.0119** |
| dCor gap sul null (target) — *ideale 0* | +0.680 ⚠️ | +0.649 | **+0.673** |
| dCor gap sul null (source) — *ideale 0* | +0.693 ⚠️ | +0.647 | **+0.763** |
| Probe lineare su e^c — *ideale 50%* | 92.5% ⚠️ | 98.82% | **98.76%** |
| Probe MLP su e^c — *ideale 50%* | 90.4% ⚠️ | 99.13% | **99.77%** |
| Disent. score lineare — *ideale >35%* | 0.99% | 1.18% | **1.24%** |
| Disent. score MLP — *ideale >35%* | −0.72% | 0.87% | **0.23%** |

⚠️ CDs→Instruments ha 1.842 utenti sovrapposti contro i 5.000 campionati altrove. Il null dipende da quel numero, quindi quella colonna **non è confrontabile** con le altre due. Le due direzioni Elec/Cloth sì.

Confrontiamo le due direzioni confrontabili. Decuplicando il peso della regolarizzazione, l'ortogonalità **migliora di tre volte**: coseno da 0.036 a 0.012. La loss fa esattamente il mestiere per cui è scritta.

L'indipendenza reale **non migliora affatto**:

- probe lineare fermo: 98.82% → 98.76%, differenza nel rumore;
- probe MLP leggermente peggio: 99.13% → 99.77%;
- gap di dCor peggio, soprattutto sul lato source: +0.647 → +0.763.

I probe sono a soffitto: canale specific al 100%, shared sopra il 98%. Lì non c'è più spazio per muoversi, quindi la misura informativa è il gap di dCor — l'unica che cambia in modo apprezzabile. E cambia nella direzione sbagliata.

**Conclusione: la manopola dell'ortogonalità non compra indipendenza a nessun prezzo.** Decuplicarla dà esattamente ciò che la loss penalizza, vettori più perpendicolari, e zero progresso su ciò che serviva. Nel run con la regolarizzazione più forte, il canale che dovrebbe aver dimenticato il dominio è identificabile al **99,77%**.

**L'analogia.** Prendi dei punti su una circonferenza. Le coordinate x e y hanno correlazione zero: sono "perpendicolari" in senso statistico. Eppure conoscendo x sai quanto vale y, a meno del segno. Sono **scorrelate ma dipendenti**. Il modello azzera il prodotto scalare e ottiene precisamente quello. Non l'indipendenza, che era l'obiettivo.

**Cosa significa.** Le metriche con cui l'area valuta il disentanglement possono dare ottimi risultati mentre l'obiettivo resta mancato. È un problema di metodo, non di questa implementazione.

> **Confondente da dichiarare.** Tra i due run grandi non è cambiato solo `cl_org_weight` (0.1 → 1). Anche `cl_sim_weight` e `item_cl_weight` sono scesi da 0.1 a 0.01. `cl_sim_weight` è la loss che **allinea** le common features tra domini: indebolita di dieci volte, può benissimo lasciare più informazione di dominio nel canale shared. Con i dati attuali questa spiegazione vale quanto la mia. E due run sono due punti: non fanno un andamento.
>
> **Effetto soffitto.** Con i probe sopra il 98% in entrambe le direzioni, il confronto avviene dove lo strumento non ha più risoluzione. Lo sweep va quindi valutato sul gap di dCor, o con un probe deliberatamente più debole che non saturi.

### Risultato 2 — τ misura qualcosa di reale, e riconosce il collasso

Sul dataset piccolo τ era una costante: 0.4974, deviazione standard **0.006**, uguale per ogni utente e ogni prodotto. Sembrava un bug del mio codice. Non lo era: lì i due canali erano **lo stesso vettore** (coseno 0.997), quindi contribuivano per forza allo stesso modo. τ diceva la verità.

Il meccanismo del collasso: entrambi i canali si ottengono filtrando lo stesso vettore con due "rubinetti" appresi, e i rubinetti avevano imparato valori quasi identici, 0.408 contro 0.409. Da lì degenera tutto, compresi i pesi dell'attenzione, che finiscono a 0.4999 contro 0.5001.

| | CDs→Instruments | Elec→Cloth | Cloth→Elec |
|---|---|---|---|
| τ pooled | 0.4967 | 0.4245 | 0.5031 |
| deviazione standard | **0.006** | 0.084 | 0.080 |
| τ per utente (min–max) | ~0.50 piatto | 0.242 – 0.741 | 0.310 – 0.733 |
| transfer-driven (τ≥0.6) | 0% | 3.0% | 11.2% |
| native-driven (τ≤0.4) | 0% | 39.0% | 7.8% |

**Due situazioni opposte dietro lo stesso numero.** τ vale ~0.50 sia su CDs→Instruments sia su Cloth→Elec. Ma nel primo caso è un collasso — deviazione 0.006, tutti identici. Nel secondo è una distribuzione vera centrata a metà — deviazione 0.080, utenti da 0.31 a 0.73. **Chi guarda solo la media conclude il contrario del vero: il segnale è la dispersione.**

**Cosa significa.** L'attribuzione distingue raccomandazioni di natura diversa. E τ funziona come **spia diagnostica**: se è piatta, il disentanglement è collassato. È leggibile a colpo d'occhio e, a differenza delle metriche esistenti, è definita sulla singola raccomandazione invece che sull'intero spazio latente.

### Risultato 3 — Il transfer va ai clienti abituali, non ai nuovi (replica debole)

L'assunzione comune è che il cross-domain serva a chi ha poca storia nel dominio target: non sapendo cosa consigliargli, il sistema attinge dall'altro dominio. È la motivazione con cui questi modelli vengono giustificati.

In entrambi i run grandi succede il contrario: **più storia ha l'utente nel dominio target, più la raccomandazione dipende dal transfer.**

| | Elec→Cloth | Cloth→Elec |
|---|---|---|
| corr(τ, storia target) — Pearson | **+0.41** | **+0.11** |
| corr(τ, storia target) — Spearman | +0.25 | +0.10 |
| τ medio, storia 5–19 | 0.4145 | 0.5010 |
| τ medio, storia 20+ | 0.4813 | 0.5231 |

Su Elec→Cloth ho escluso la spiegazione banale, cioè che siano utenti molto attivi ovunque. La correlazione con la storia nel dominio *source* è praticamente zero (+0.06), le due storie sono poco correlate tra loro (0.15), e controllando per la storia source l'effetto resta identico (+0.42) in ogni fascia.

Su Cloth→Elec il segno **si replica**, ma l'effetto è quattro volte più debole.

**Cosa significa.** La direzione è consistente su due run indipendenti, quindi difficilmente è un artefatto. L'intensità no, e va trattata come non stabilita. Se regge, ribalta la narrazione standard: non è che il modello presta conoscenza a chi ne ha bisogno, ma che serve una base locale sufficiente **prima** che il transfer possa agganciarsi a qualcosa.

---

## 4. Come vanno letti insieme

I risultati 1 e 3 sono in tensione.

Il risultato 3 assume che il canale "shared" contenga conoscenza trasferita. Il risultato 1 dice che quel canale resta identificabile per dominio al 98–99%. Quindi:

- come **misura**, il risultato 3 è solido: i numeri sono quelli, il confondente principale è escluso;
- come **interpretazione**, dipende dal fidarsi dell'etichetta "shared". Al momento non dovremmo.

Per questo, come contributo scientifico **il risultato 1 vale più del risultato 3**: il primo dice qualcosa su come l'area valuta sé stessa, il secondo vale finché l'architettura fa quello che dichiara.

Un dettaglio da tenere d'occhio: il run con il disentanglement **peggiore** (Cloth→Elec, probe al 99,8%) è quello che **raccomanda meglio** (Recall@20 0.0397 contro 0.0253). Non ne trarrei conclusioni — cambia il dominio target, e quindi la difficoltà. Ma se reggesse a un confronto controllato, direbbe che il disentanglement non è ciò che fa funzionare il modello.

---

## 5. Limiti

- **Un checkpoint per configurazione, un seed.** Niente è stato replicato.
- I confronti cambiano **più variabili insieme**: direzione, dominio target, tre pesi di loss su quattro. Nessun confronto qui è a variabile singola.
- I dataset sono 10-core: **nessun utente davvero freddo**. Il risultato 3 riguarda utenti già attivi e non dice nulla sul cold-start.
- τ descrive **come è composto** il punteggio, non cosa succederebbe togliendo un canale e ricalcolando la classifica.
- τ alto **non** significa raccomandazione migliore. È un'attribuzione, non una valutazione.
- La scomposizione esatta richiede `fuse_mode='attention'`.
- Solo gli utenti sovrapposti hanno un τ definito.
- I probe sono **a soffitto** (98–100%) nei due run grandi: lì non distinguono più configurazioni diverse.
- Il gap di dCor dipende dal numero di utenti campionati: confrontabile solo a `--dcor_sample` uguale. CDs→Instruments resta fuori confronto.
- **Previsione sbagliata**, annotata per onestà: mi aspettavo τ più basso su Cloth→Elec. È uscito più alto (0.503 contro 0.425), con più raccomandazioni transfer-driven. Non avevo una buona ragione per aspettarmelo.
- **Affermazione ritirata**: avevo scritto che *ogni* misura di indipendenza peggiora aumentando `cl_org_weight`. Con lo strumento corretto il probe lineare è invariato (98.82% → 98.76%), e solo il gap di dCor peggiora. La conclusione giusta è che l'ortogonalità non compra indipendenza, non che la distrugga.

---

## 6. Prossimi passi

**1. Sweep controllato su `cl_org_weight`.** È l'esperimento che chiude il risultato 1. Una direzione sola, tutto fisso, solo quel peso su `{0.01, 0.1, 1, 10}`. Quattro run da ~1h35m. Da valutare sul **gap di dCor**, non sui probe, che saturano. Se il gap non scende mentre il coseno crolla, il risultato è dimostrato e non attaccabile — ed è il contributo principale del paper. Oggi resta suggestivo ma confondato.

**2. Replica con più seed**, per capire quanto sono stabili τ e la correlazione con la storia. Serve soprattutto dopo che il risultato 3 è sceso da +0.41 a +0.11.

**3. Test sulla forma della loss di ortogonalità.** Penalizza il prodotto scalare grezzo, che resta piccolo quando i vettori sono piccoli, a prescindere dall'angolo. Una versione normalizzata potrebbe comportarsi diversamente.

**4. Controfattuale.** Azzerare il canale shared, ricalcolare le classifiche, e passare dall'attribuzione alla causalità.

**5. Effetto della semantic loss.** `semantic_loss_weight` tira il canale shared degli item verso l'embedding testuale del prodotto. Ma il testo di un prodotto dice chiaramente di che dominio è, quindi quella loss potrebbe iniettare informazione di dominio proprio dove non dovrebbe essercene. Nei tre run era spenta, quindi non spiega nulla di quanto sopra. Accenderla e misurare il probe costa quasi zero.
