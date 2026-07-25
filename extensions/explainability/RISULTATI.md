# Attribuzione esatta in DGCDR — cosa fa la pipeline e cosa abbiamo trovato

Documento di lavoro, aggiornato al 25 luglio 2026. Branch `explainability-1`.
Tre run analizzati: CDs→Instruments, Elec→Cloth, Cloth→Elec.

---

## 1. Cosa fa la pipeline, in parole semplici

DGCDR divide i gusti di ogni utente in due parti:

- **shared** — i gusti che l'utente porta con sé dall'altro dominio (es. dall'abbigliamento all'elettronica);
- **specific** — i gusti che valgono solo nel dominio in cui gli stiamo raccomandando qualcosa.

Poi le rimette insieme e produce un punteggio per ogni prodotto. La pipeline che ho costruito fa una cosa sola, ma la fa in modo verificabile: **prende il punteggio finale e lo riapre, dicendo quanta parte viene da ciascuna delle due parti**.

Da questa scomposizione nasce una quantità che chiamo **τ (transfer ratio)**: un numero tra 0 e 1 che dice, per ogni singola raccomandazione, quanto quella raccomandazione dipende da conoscenza importata dall'altro dominio invece che dai gusti locali dell'utente.

- τ vicino a 1 → "ti consiglio questo perché ho capito i tuoi gusti guardando l'altro dominio"
- τ vicino a 0 → "ti consiglio questo per come ti sei comportato qui"
- τ ≈ 0.5 → le due cose pesano uguale

### Perché non è il solito explainer

Quasi tutti i sistemi che "spiegano" una raccomandazione con un LLM funzionano così: danno all'LLM la cronologia dell'utente e gli chiedono di inventare una motivazione plausibile. Il testo che esce suona bene, ma **nessuno garantisce che abbia a che fare con quello che il modello ha realmente calcolato**. È una ricostruzione a posteriori.

Qui è diverso, e il motivo è aritmetico. DGCDR combina le sue componenti con una somma, e il punteggio è un prodotto scalare. Quindi il punteggio **si spacca esattamente**, senza approssimazioni:

```
punteggio = (contributo shared) + (contributo specific) + (contributo collaborativo grezzo)
```

Non è una stima, non è un modello surrogato come LIME o SHAP: è la stessa aritmetica che il modello ha eseguito, riscritta. Per essere sicuri che non ci siano errori, ogni esecuzione **ricompone i pezzi e li confronta con l'output vero del modello**, e si rifiuta di produrre spiegazioni se non tornano. Sui tre run l'errore relativo è stato tra 1e-07 e 3e-07, cioè puro arrotondamento in virgola mobile.

L'LLM, quando lo si usa, arriva **dopo** e ha un ruolo molto più modesto: riceve i numeri già calcolati e li mette in italiano corrente, con il vincolo esplicito di non contraddirli. Non decide lui il perché.

### I pezzi

| file | cosa fa |
|---|---|
| `channels.py` | riapre gli embedding nelle componenti e verifica che tornino |
| `attribution.py` | calcola la matrice dei contributi e τ |
| `metadata.py` | traduce gli ID dei prodotti in titoli leggibili (opzionale) |
| `verbalize.py` | costruisce il prompt vincolato e parla con l'LLM (opzionale) |
| `explain_dgcdr.py` | lo script da lanciare |
| `test_decomposition.py` | autotest: 8 controlli, da rilanciare su ogni nuovo modello |

La pipeline **non tocca il codice del modello**: ricostruisce tutto usando le funzioni pubbliche di DGCDR. Il training resta esattamente com'era.

### Quando si rifiuta di funzionare

Meglio un errore che una spiegazione falsa. Lo script si blocca se il modello usa una fusione non additiva (`fuse_mode='concat'`), se il disentanglement è disattivato, o se il modello non è in modalità valutazione. In tutti questi casi la scomposizione non sarebbe esatta, e una spiegazione "quasi giusta" è peggio di nessuna spiegazione.

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

Il primo era nato come prova tecnica per validare il codice; gli altri due sono gli esperimenti veri, sulle due direzioni della stessa coppia di domini.

---

## 3. I tre risultati

### Risultato 1 — Ortogonalità e indipendenza si muovono in direzioni opposte

È il risultato principale, e il terzo run lo ha rafforzato parecchio.

Il modello viene addestrato a rendere i due canali **perpendicolari**, cosa che misuriamo con il coseno. Ma l'obiettivo scientifico non è la perpendicolarità: è che il canale *shared* abbia davvero dimenticato da quale dominio viene. Questo si misura in altro modo — con la distance correlation e chiedendo a un classificatore di indovinare il dominio guardando solo quel canale.

| | CDs→Instruments | Elec→Cloth | **Cloth→Elec** |
|---|---|---|---|
| `cl_org_weight` | 1 | 0.1 | **1** |
| Coseno (target) — *ideale 0* | 0.9973 | 0.0365 | **0.0119** |
| dCor (target) — *ideale 0* | 0.9989 | 0.7434 | **0.7807** |
| Probe lineare su e^c — *ideale 50%* | 85.4% | 81.8% | **85.8%** |
| Probe MLP su e^c — *ideale 50%* | 93.3% | 98.7% | **99.8%** |
| Disent. score MLP — *ideale >35%* | −0.27% | 1.34% | **0.24%** |

Confrontando le due direzioni: decuplicando il peso della regolarizzazione, l'ortogonalità **migliora di tre volte** (coseno da 0.036 a 0.012) — la loss fa esattamente il suo mestiere. E **ogni singola misura di indipendenza reale peggiora**: dCor sale, entrambi i probe salgono, il punteggio di disentanglement scende.

Non è solo che spingere di più non aiuta. Spingere di più fa danno. Nel run con la regolarizzazione più forte il canale che dovrebbe aver dimenticato il dominio è identificabile al **99,76%**, praticamente perfetto.

**L'analogia.** Prendi dei punti disposti su una circonferenza. La coordinata x e la y hanno correlazione zero, sono "perpendicolari" in senso statistico. Eppure conoscendo la x sai esattamente quanto vale la y a meno del segno. Sono **scorrelate ma tutt'altro che indipendenti**. Il modello ottimizza una funzione che azzera il prodotto scalare, e ottiene precisamente quello: prodotto scalare zero. Non l'indipendenza, che era ciò che serviva.

**Cosa significa.** Le metriche con cui la comunità valuta il disentanglement possono dare risultati ottimi mentre l'obiettivo resta mancato — e peggiorare proprio mentre migliorano. È un problema metodologico che riguarda l'area, non un difetto di questa implementazione.

> **Il confondente da dichiarare.** Tra i due run grandi non è cambiato solo `cl_org_weight` (0.1 → 1). Anche `cl_sim_weight` e `item_cl_weight` sono scesi da 0.1 a 0.01. `cl_sim_weight` è la loss che **allinea** le common features tra i domini: indebolendola di dieci volte, è del tutto plausibile che il canale shared trattenga più informazione di dominio. Questa spiegazione alternativa è, con i dati attuali, altrettanto valida della mia. E due run sono due punti: non stabiliscono un andamento. Vedi la sezione 6, esperimento 1.

### Risultato 2 — Lo strumento misura qualcosa di reale, e distingue il collasso

Sul dataset piccolo τ era una costante: 0.4974 con deviazione standard **0.006**, identica per ogni utente e ogni prodotto. Sembrava un bug. Non lo era: lì i due canali erano letteralmente **lo stesso vettore** (coseno 0.997), quindi contribuivano per forza in parti uguali. τ stava dicendo la verità.

Ho ricostruito il meccanismo del collasso: entrambi i canali si ottengono filtrando lo stesso vettore di partenza con due "rubinetti" appresi, e i due rubinetti avevano imparato valori quasi identici (0.408 contro 0.409). Da lì degenera tutto a catena, anche i pesi dell'attenzione, che finiscono a 0.4999 contro 0.5001.

| | CDs→Instruments | Elec→Cloth | Cloth→Elec |
|---|---|---|---|
| τ pooled | 0.4967 | 0.4245 | 0.5031 |
| deviazione standard | **0.006** | 0.084 | 0.080 |
| τ per utente (min–max) | ~0.50 piatto | 0.242 – 0.741 | 0.310 – 0.733 |
| transfer-driven (τ≥0.6) | 0% | 3.0% | 11.2% |
| native-driven (τ≤0.4) | 0% | 39.0% | 7.8% |

**Attenzione a non confondere due cose diverse.** Su CDs→Instruments τ vale ~0.50; su Cloth→Elec vale ~0.50. Ma nel primo caso è un collasso (deviazione 0.006: *tutti* uguali), nel secondo è una distribuzione vera centrata sulla metà (deviazione 0.080, utenti da 0.31 a 0.73). Il valore medio da solo non dice nulla: **è la dispersione a distinguere un modello che bilancia i canali da un modello che non li ha separati affatto.**

**Cosa significa.** Due cose. L'attribuzione distingue davvero raccomandazioni di natura diversa. E τ funziona come **spia diagnostica**: se è piatta, il disentanglement è collassato. È un test leggibile a colpo d'occhio e, a differenza delle metriche esistenti, è definito sulla singola raccomandazione invece che sull'intero spazio latente.

### Risultato 3 — Il transfer va ai clienti abituali, non ai nuovi (replica debole)

Ci si aspetta che il cross-domain serva soprattutto a chi ha poca storia nel dominio target: non sapendo cosa consigliargli, il sistema attinge dall'altro dominio. È la motivazione con cui questi modelli vengono normalmente giustificati.

In entrambi i run grandi succede il contrario: **più un utente ha storia nel dominio target, più la sua raccomandazione dipende dal transfer.**

| | Elec→Cloth | Cloth→Elec |
|---|---|---|
| corr(τ, storia target) — Pearson | **+0.41** | **+0.11** |
| corr(τ, storia target) — Spearman | +0.25 | +0.10 |
| τ medio, storia 5–19 | 0.4145 | 0.5010 |
| τ medio, storia 20+ | 0.4813 | 0.5231 |

Su Elec→Cloth avevo escluso la spiegazione banale — che siano utenti molto attivi ovunque: la correlazione con la storia nel dominio *source* era praticamente zero (+0.06), le due storie sono poco correlate tra loro (0.15), e controllando statisticamente per la storia source l'effetto restava identico (+0.42) in ogni fascia.

Su Cloth→Elec il segno si **replica**, ma l'effetto è circa quattro volte più debole.

**Cosa significa.** La direzione dell'effetto è consistente su due run indipendenti, quindi difficilmente è un artefatto. La sua *intensità* però non è stabile, e va trattata come non stabilita. Se regge, mette in discussione la narrazione standard sul perché il cross-domain funziona: non è che il modello presta conoscenza a chi ne ha bisogno, ma che serve una base di dati locale sufficiente **prima** che il transfer possa agganciarsi a qualcosa.

---

## 4. Come vanno letti insieme

I risultati 1 e 3 sono in tensione, ed è importante non nasconderlo.

Il risultato 3 dà per buono che il canale "shared" rappresenti conoscenza trasferita. Ma il risultato 1 dice che quel canale resta identificabile per dominio al 98–99% in entrambi i run grandi. Quindi:

- come **misura**, il risultato 3 è solido: i numeri sono quelli e il confondente principale è stato escluso;
- come **interpretazione**, dipende da quanto ci fidiamo dell'etichetta "shared", e in questo momento non dovremmo fidarcene molto.

È anche il motivo per cui, come contributo scientifico, **il risultato 1 vale più del risultato 3**: il primo dice qualcosa sul modo in cui l'area valuta sé stessa, il secondo è un'osservazione empirica che vale finché l'architettura fa quello che dichiara.

Una nota che vale la pena tenere d'occhio: il run con il disentanglement **peggiore** (Cloth→Elec, probe al 99,8%) è anche quello che **raccomanda meglio** (Recall@20 0.0397 contro 0.0253). Non ne trarrei conclusioni, perché cambia il dominio target e quindi la difficoltà del task. Ma se reggesse a un confronto controllato, direbbe che il disentanglement non è ciò che fa funzionare il modello.

---

## 5. Limiti da dichiarare

- **Un solo checkpoint per configurazione, un solo seed.** Nulla è stato replicato con seed diversi.
- I confronti tra run cambiano **più variabili insieme**: direzione del transfer, dominio target, e tre pesi di loss su quattro. Nessuno dei confronti qui è a variabile singola.
- I dataset sono 10-core: **non contengono utenti davvero freddi**. Il risultato 3 riguarda utenti già attivi e non dice nulla sul cold-start vero.
- τ descrive **come è composto** il punteggio, non cosa succederebbe togliendo un canale e ricalcolando la classifica. Quel controfattuale è una domanda diversa.
- τ alto **non** significa che la raccomandazione sia migliore grazie al transfer. È un'attribuzione, non una valutazione.
- La scomposizione esatta richiede `fuse_mode='attention'`. È la configurazione dei setting del paper, ma resta una restrizione.
- Solo gli utenti sovrapposti tra i due domini hanno un τ definito.
- Una previsione sbagliata, annotata per onestà: mi aspettavo che su Cloth→Elec τ fosse più schiacciato verso il basso. È risultato invece più alto (0.503 contro 0.425) e con più raccomandazioni transfer-driven. Non avevo una buona ragione per quell'aspettativa.

---

## 6. Prossimi passi, in ordine di priorità

**1. Sweep controllato su `cl_org_weight`.** È l'esperimento che scioglie il nodo del risultato 1. Una sola direzione, tutto fisso, solo quel peso che varia su `{0.01, 0.1, 1, 10}`. Quattro run da ~1h35m. Se il probe MLP sale monotonicamente mentre il coseno scende, il risultato è dimostrato in modo pulito e non attaccabile — ed è il contributo principale del paper. Con i dati attuali resta un'osservazione suggestiva ma confondata.

**2. Replica con più seed**, per capire quanto sono stabili τ e la correlazione con la storia — soprattutto dopo che il risultato 3 si è indebolito da +0.41 a +0.11.

**3. Verifica dell'ipotesi sulla forma della loss di ortogonalità**: penalizza il prodotto scalare grezzo, che resta piccolo quando i vettori sono piccoli, indipendentemente dall'angolo. Una versione normalizzata potrebbe comportarsi diversamente.

**4. Controfattuale**: azzerare il canale shared e ricalcolare le classifiche, per passare dall'attribuzione alla causalità.

**5. Effetto della semantic loss.** `semantic_loss_weight` tira il canale shared degli item verso l'embedding testuale del prodotto. Ma il testo di un prodotto è fortemente indicativo del dominio, quindi quella loss potrebbe iniettare informazione di dominio proprio nel canale che dovrebbe esserne privo. Nei tre run era disattivata, quindi non spiega nulla di quanto sopra — ma accenderla e misurare il probe è un esperimento a costo quasi zero.
