# Attribuzione esatta in DGCDR — cosa fa la pipeline e cosa abbiamo trovato

Documento di lavoro, 25 luglio 2026. Branch `explainability-1`.

---

## 1. Cosa fa la pipeline, in parole semplici

DGCDR divide i gusti di ogni utente in due parti:

- **shared** — i gusti che l'utente porta con sé dall'altro dominio (es. dalla musica agli strumenti musicali);
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

Non è una stima, non è un modello surrogato come LIME o SHAP: è la stessa aritmetica che il modello ha eseguito, riscritta. Per essere sicuri che non ci siano errori, ogni esecuzione **ricompone i pezzi e li confronta con l'output vero del modello**, e si rifiuta di produrre spiegazioni se non tornano. Nella pratica l'errore è dell'ordine di 1e-07, cioè puro arrotondamento dei numeri in virgola mobile.

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

## 2. I due esperimenti

| | CDs → Instruments | Elec → Cloth |
|---|---|---|
| utenti | 1.842 | 35.827 |
| prodotti (target) | 3.610 | 72.669 |
| interazioni (dominio target) | ~16 mila | ~847 mila |
| iperparametri | default del repo | quelli del paper (sez. 2.1) |
| `cl_org_weight` | 1 | 0.1 |
| convergenza | epoca 159 | epoca 84 |
| Recall@20 (test) | 0.0648 | 0.0253 |

Il primo era nato come prova tecnica per validare il codice, il secondo è l'esperimento vero.

> **Attenzione a un confondente.** I due run differiscono per *due* cose insieme: la dimensione del dataset **e** gli iperparametri. Non posso quindi attribuire le differenze qui sotto solo alla dimensione. Anzi, c'è un dettaglio che merita attenzione: il dataset piccolo aveva la forza della regolarizzazione di ortogonalità **dieci volte più alta** e ha ottenuto il risultato peggiore. Qualunque cosa faccia la differenza, non è il peso di quella loss.

---

## 3. I tre risultati

### Risultato 1 — La separazione dei canali è apparente, non reale

È il risultato più importante.

Sul dataset grande la divisione tra shared e specific **sembra** riuscita: i due vettori risultano perpendicolari (coseno 0.036, dove 0 è l'ideale). E perpendicolari è esattamente il criterio che il modello viene addestrato a rispettare.

Ma se prendo **solo** il canale shared — quello che dovrebbe contenere gusti universali, validi in entrambi i domini — e chiedo a un classificatore *"da quale dominio viene questo vettore?"*, quello indovina nel **98,7% dei casi**. Il canale che per costruzione dovrebbe aver dimenticato il dominio se lo ricorda quasi perfettamente.

L'analogia esatta è questa: prendi dei punti disposti su una circonferenza. La coordinata x e la coordinata y hanno correlazione zero, sono "perpendicolari" in senso statistico. Eppure conoscendo la x sai esattamente quanto vale la y a meno del segno. Sono **scorrelate ma tutt'altro che indipendenti**.

Il modello ottimizza una funzione che azzera il prodotto scalare tra i due canali, e ottiene precisamente quello: prodotto scalare zero. Non l'indipendenza, che era ciò che serviva davvero.

**Cosa significa.** Le metriche con cui la comunità valuta il disentanglement — coseno, ortogonalità — possono dare risultati ottimi mentre l'obiettivo scientifico resta mancato. Serve guardare misure di dipendenza vera (dCor, che qui vale 0.74 contro un ideale di 0) e probe non lineari. È un problema metodologico che riguarda l'area, non un difetto di questa implementazione.

### Risultato 2 — Lo strumento misura qualcosa di reale

Sul dataset piccolo τ era una costante: 0.4974 con deviazione standard 0.006, identica per ogni utente e ogni prodotto. Sembrava un bug del mio codice. Non lo era: lì i due canali erano letteralmente **lo stesso vettore** (coseno 0.997), quindi contribuivano per forza in parti uguali. τ stava dicendo la verità.

Ho ricostruito il meccanismo del collasso: entrambi i canali si ottengono filtrando lo stesso vettore di partenza con due "rubinetti" appresi, e i due rubinetti avevano imparato valori quasi identici (0.408 contro 0.409). Da lì in poi tutto degenera a catena, anche i pesi dell'attenzione, che finiscono a 0.4999 contro 0.5001.

Sul dataset grande τ ha invece una distribuzione vera: va da 0.24 a 0.74 tra gli utenti, con il 39% delle raccomandazioni guidate dai gusti locali contro il 3% guidate dal transfer.

**Cosa significa.** Due cose. Primo, l'attribuzione distingue davvero raccomandazioni di natura diversa. Secondo — e forse più utile — **τ funziona come spia diagnostica**: se resta inchiodata a 0.5, il disentanglement è collassato. È un test leggibile a colpo d'occhio, e a differenza delle metriche esistenti è definito sulla singola raccomandazione invece che sull'intero spazio latente.

### Risultato 3 — Il transfer va ai clienti abituali, non ai nuovi

Questo ribalta un'assunzione diffusa.

Ci si aspetta che il cross-domain serva soprattutto a chi ha poca storia nel dominio target: non sapendo cosa consigliargli, il sistema attinge dall'altro dominio. È la motivazione con cui questi modelli vengono normalmente giustificati.

Qui succede il contrario: **più un utente ha storia nel dominio target, più la sua raccomandazione dipende dal transfer.** La correlazione è +0.41.

Ho controllato la spiegazione banale — che siano semplicemente utenti molto attivi ovunque, e quindi con più dati da entrambe le parti — e non regge:

- la correlazione tra τ e la storia nel dominio **source** è praticamente zero (+0.06);
- la storia nei due domini è poco correlata tra loro (0.15), quindi non sono la stessa cosa mascherata;
- controllando statisticamente per la storia source, l'effetto resta **identico** (+0.42);
- confrontando utenti con la stessa quantità di storia source, l'effetto compare in **ogni** fascia.

L'effetto è concentrato agli estremi: τ resta piatta intorno a 0.41 per tre quarti degli utenti e sale a 0.46–0.47 solo nel quarto più attivo.

**Cosa significa.** Se regge alla replica, mette in discussione la narrazione standard sul perché il cross-domain funziona. Non è che il modello "presta" conoscenza a chi ne ha bisogno: sembra piuttosto che serva una base di dati locale sufficiente **prima** che il transfer possa agganciarsi a qualcosa.

---

## 4. Come vanno letti insieme

I risultati 1 e 3 sono in tensione, ed è importante non nasconderlo.

Il risultato 3 dà per buono che il canale "shared" rappresenti conoscenza trasferita. Ma il risultato 1 dice che quel canale è ancora identificabile per dominio al 98,7%. Quindi:

- come **misura**, il risultato 3 è solido: i numeri sono quelli, il confondente è stato escluso;
- come **interpretazione**, dipende da quanto ci fidiamo dell'etichetta "shared", e in questo momento non dovremmo fidarcene molto.

È anche il motivo per cui, come contributo scientifico, **il risultato 1 vale più del risultato 3**: il primo dice qualcosa sul modo in cui l'area valuta sé stessa, il secondo è un'osservazione empirica che vale finché l'architettura fa quello che dichiara.

---

## 5. Limiti da dichiarare

- **Un solo checkpoint per dataset, un solo seed, una sola direzione.** Nulla di quanto sopra è stato replicato. Va fatto prima di scriverlo in un paper.
- Il dataset Elec/Cloth è 10-core: **non contiene utenti davvero freddi**. Il risultato 3 riguarda utenti già attivi e non dice nulla sul cold-start vero.
- τ descrive **come è composto** il punteggio, non cosa succederebbe togliendo un canale e ricalcolando la classifica. Quel controfattuale è una domanda diversa.
- τ alto **non** significa che la raccomandazione sia migliore grazie al transfer. È un'attribuzione, non una valutazione.
- La scomposizione esatta richiede `fuse_mode='attention'`. È la configurazione dei setting del paper, ma resta una restrizione.
- Solo gli utenti sovrapposti tra i due domini hanno un τ definito.
- Il confondente dimensione/iperparametri tra i due esperimenti descritto sopra.

---

## 6. Prossimi passi

1. **Cloth → Elec**, la direzione opposta, dove `cl_org_weight` è dieci volte più forte. Se lì il probe non lineare scende, qualità del disentanglement e interpretabilità di τ si muovono insieme e i risultati 1 e 3 diventano una relazione invece che due osservazioni isolate.
2. **Replica** con più seed, per capire quanto sono stabili τ e la correlazione con la storia.
3. **Verifica dell'ipotesi sulla loss di ortogonalità**: penalizza il prodotto scalare grezzo, che resta piccolo quando i vettori sono piccoli, indipendentemente dall'angolo. Una versione normalizzata potrebbe comportarsi diversamente. Da testare, per ora è solo un sospetto.
4. **Controfattuale**: azzerare il canale shared e ricalcolare le classifiche, per passare dall'attribuzione alla causalità.
