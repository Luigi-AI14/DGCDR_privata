# Cosa abbiamo fatto e cosa abbiamo trovato

Documento di lavoro, aggiornato al 28 luglio 2026. Branch `explainability-2`.
Sei checkpoint analizzati: CDs→Instruments, Elec→Cloth, Cloth→Elec a tre valori
di `cl_org_weight`, e Movie→Book su Douban.

---

## 1. Il problema di partenza

DGCDR raccomanda usando due domini. Guarda il comportamento dell'utente nel
dominio **source** per raccomandare meglio nel dominio **target**.

Per farlo divide le preferenze in due parti:

- **shared** — la parte che vale in entrambi i domini;
- **specific** — la parte che vale solo nel dominio target.

L'idea è che la parte shared sia quella che trasferisce conoscenza da un dominio
all'altro.

**L'obiettivo del lavoro è spiegare le raccomandazioni cross-domain**: dire, per
ogni item raccomandato, quanto dipende da conoscenza importata dall'altro dominio
e quanto dalle preferenze native.

Da qui discende tutto il resto. Una spiegazione del genere ha senso solo se i due
canali rappresentano davvero ciò che il loro nome promette. Verificarlo non è una
deviazione dall'obiettivo: **è il presupposto perché la spiegazione sia
affidabile**, ed è ciò che ci ha portati alle verifiche della sezione 5.

---

## 2. Lo strumento che abbiamo costruito

Il modello calcola un punteggio per ogni item e raccomanda quelli col punteggio
più alto. Abbiamo costruito uno strumento che **decompone quel punteggio**:

```
punteggio = (contributo del canale shared) + (contributo del canale specific) + (contributo collaborativo grezzo)
```

Da qui nasce il **transfer ratio τ**, tra 0 e 1:

- **τ vicino a 1** → il punteggio viene soprattutto dal canale shared
- **τ vicino a 0** → viene soprattutto dal canale specific
- **τ ≈ 0.5** → i due canali pesano uguale

### Perché la decomposizione è affidabile

Gli explainer basati su LLM ricevono la cronologia dell'utente e generano una
motivazione plausibile. Il testo suona bene, ma niente garantisce che
corrisponda a quello che il modello ha calcolato.

Qui il meccanismo è aritmetico. DGCDR fonde i canali con una somma e il punteggio
è un prodotto scalare, quindi la decomposizione è **esatta**. Non è
un'approssimazione e non è un surrogato come LIME o SHAP.

Ogni esecuzione ricompone i canali e li confronta con l'output vero del modello.
Se non coincidono, si ferma. Sui sei checkpoint l'errore relativo sta tra
1e-07 e 3e-07: solo arrotondamento in virgola mobile.

---

## 3. La verifica che regge tutto: riproduciamo il paper

Prima di criticare un modello bisogna averlo ricostruito bene. Confronto con la
Tabella 3 dell'articolo:

| | | paper | nostro | scarto |
|---|---|---|---|---|
| **Elec→Cloth** | Recall@20 | .0260 | .0253 | −2.7% |
| | NDCG@20 | .0173 | .0173 | **0%** |
| **Cloth→Elec** | Recall@20 | .0403 | .0397 | −1.5% |
| | NDCG@20 | .0247 | .0244 | −1.2% |
| **Movie→Book** | Recall@20 | .1369 | .1341 | −2.0% |
| | MRR@20 | .1557 | .1528 | −1.9% |
| | HR@20 | .5505 | .5424 | −1.5% |
| | NDCG@20 | .0954 | .0934 | −2.1% |

Scarti tra l'1% e il 3% su **due coppie di domini molto diverse** — Amazon
(sparsa, 35k utenti) e Douban (densa, 10k utenti e 2,3 milioni di interazioni).
È normale variabilità da seed. **Tutto quello che segue riguarda il modello
pubblicato**, non una nostra versione difettosa.

---

## 4. Prima sorpresa: lo strumento di misura era difettoso

`evaluate_disentanglement.py` misura la qualità della separazione. Aveva tre
difetti, e **tutti e tre gonfiavano il risultato**.

1. **I probe giravano su feature non standardizzate.** I valori stanno intorno a
   0.02, quindi la penalità L2 di default dominava e i classificatori andavano in
   underfitting. Nessun warning. Il probe lineare passa da 81.79% a **98.82%**
   una volta scalate le feature: diciassette punti di sottostima.
2. **dCor era confrontato con uno zero irraggiungibile.** È l'estimatore
   V-statistic, distorto verso l'alto a N finito e alta dimensione: due gaussiane
   indipendenti con N=5000 e D=256 danno 0.41. Ora lo script permuta le righe di
   uno spazio per ottenere il null empirico e riporta il gap.
3. **19 GB di memoria** per la matrice delle distanze a 35k utenti. Ora
   sotto-campiona: a 5.000 utenti la stima cambia di un centesimo.

**Questo è già un risultato.** Con il difetto 1, Elec→Cloth sembrava avere un
disentanglement score del 18%. Il valore vero è 1.18%. Un probe sotto-specificato
sovrastima la separazione in silenzio, e la trappola vale per chiunque riporti
queste metriche.

---

## 5. Cosa il framework ha rivelato sul modello

Questa sezione è la validazione del contributo. Ogni punto è una proprietà di
DGCDR che le metriche esistenti non mostravano e che il framework di
attribuzione ha reso visibile. Sono anche i motivi per cui una spiegazione
ingenua, del tipo "questo item ti è stato consigliato per i tuoi gusti
trasferiti", sarebbe fuorviante.

### 5.1 L'ortogonalità non compra indipendenza

Il modello è addestrato a rendere i due canali **ortogonali**. Il coseno misura
proprio quello.

Ma ortogonale **non vuol dire indipendente**.

> **L'esempio che chiarisce.** Prendi dei punti su una circonferenza. Le
> coordinate x e y hanno correlazione zero: sono ortogonali. Eppure conoscendo x
> sai quanto vale y, a meno del segno. Scorrelate e completamente dipendenti.

Sweep su Cloth→Elec, **una sola variabile che cambia**, tutto il resto fisso:

| `cl_org_weight` | 0.01 | 1 | 10 |
|---|---|---|---|
| Coseno (target) — *ideale 0* | 0.2164 | 0.0119 | **0.0068** |
| dCor gap sul null (target) — *ideale 0* | +0.7305 | +0.6726 | **+0.6479** |
| Probe lineare su e^c — *ideale 50%* | 99.19% | 98.76% | **99.12%** |
| Probe MLP su e^c — *ideale 50%* | 99.92% | 99.77% | **99.91%** |
| Recall@20 | 0.0395 | 0.0397 | **0.0401** |

Il peso aumenta di **mille volte**. I canali diventano **32 volte più
ortogonali**. Il gap di dCor migliora dell'11%. I probe non si muovono.

**Il conto che chiude la questione.** Il gap scende in modo circa lineare nel
logaritmo del peso: −0.029 per decade tra 0.01 e 1, −0.025 nell'ultima decade. A
questo ritmo, per portarlo a zero servirebbero oltre venti decadi, cioè un
`cl_org_weight` dell'ordine di 10²³. Non è una leva poco efficace: è inefficace
di un fattore astronomico.

**Non c'è nemmeno un trade-off da invocare.** Recall@20 resta identica. Anche con
peso 10, quando il termine di ortogonalità domina la loss, l'accuratezza non se
ne accorge. Il vincolo è **disaccoppiato** da ciò che fa funzionare il modello.

Il paper giustifica esplicitamente questa scelta: scrive di aver provato la
cosine similarity, di aver ottenuto risultati peggiori, e di aver preferito
l'ortogonalità perché *"provides a clearer separation between subspaces"*. La
separazione è chiara nello spazio, non nell'informazione.

**Lo stesso quadro su Douban**, che è una coppia di domini completamente diversa
— densa invece che sparsa, contenuti invece che e-commerce:

| Movie→Book | source | target |
|---|---|---|
| Coseno — *ideale 0* | 0.0094 | 0.0074 |
| dCor gap sul null — *ideale 0* | +0.6453 | +0.4732 |
| Probe lineare su e^c — *ideale 50%* | 99.56% | |
| Probe MLP su e^c — *ideale 50%* | 99.81% | |
| Disent. score MLP — *ideale >35%* | 0.17% | |

Canali ortogonali fino a 0.007, e riconoscibili per dominio al 99.8%. Il reperto
non dipende dal dataset.

### 5.2 Sul lato item la separazione non avviene affatto

DGCDR disentangla sia gli utenti sia gli item. Abbiamo guardato gli item, cosa
che nessuno aveva fatto: `evaluate_disentanglement.py` misura **solo i canali
utente**.

I due canali si ottengono filtrando lo stesso vettore con due gate appresi. Sul
lato item i due gate hanno imparato la stessa funzione:

| gate | cos(gate_c, gate_s) | sovrapposizione | medie |
|---|---|---|---|
| utente | 0.0505 | 4.7% | 0.506 / 0.440 |
| **item** | **0.9985** | **98.2%** | 0.849 / 0.852 |

Di conseguenza i due canali item inducono la stessa geometria. Correlazione tra
le matrici di similarità item-item, misura deterministica con null calibrato:

| | coseno(e^c, e^s) | corr. delle strutture | null |
|---|---|---|---|
| Elec→Cloth (sano) | 0.0365 | **+0.9984** | +0.0002 |
| CDs→Instr (collassato) | 0.9973 | +0.9997 | +0.0002 |

Su questa misura il modello sano è indistinguibile da quello collassato.

### 5.3 L'informazione di dominio è irremovibile per via lineare

Abbiamo rimosso iterativamente le direzioni più discriminative dal canale shared
degli utenti (INLP: si addestra una regressione logistica, si proietta via la sua
direzione, si ripete).

| iterazioni | probe lineare | probe MLP |
|---|---|---|
| 0 | 98.82% | 99.63% |
| 5 | **54.64%** | 98.16% |
| 20 | **50.07%** | 98.28% |
| 40 | 48.50% | 96.73% |

**Cinque direzioni contengono tutta l'informazione linearmente decodificabile.**
Rimuoverle porta il probe lineare al caso esatto. Il probe MLP resta al 98% anche
dopo quaranta.

**Questa è la spiegazione di tutto il resto.** L'ortogonalità è un vincolo
lineare. La loss di allineamento è un coseno. Il discriminatore avversariale che
avevamo provato era una rete piccola. Sono tutti strumenti lineari o quasi,
applicati a informazione che non è linearmente accessibile.

E c'è un avvertimento metodologico: chi valuta con un probe lineare, dopo un
intervento del genere, legge 50% e dichiara di aver risolto. L'informazione è
ancora tutta lì.

### 5.4 L'attention è quasi inerte, e la Figura 3 non è riproducibile

Il paper dedica una figura alla distribuzione dell'attention tra i due canali e
ci costruisce sopra due affermazioni interpretative.

Eq. (4) del paper è esattamente ciò che il codice implementa e ciò che abbiamo
misurato:

```
[a_c, a_s] = softmax( e_g · [e_c, e_s] / √d )
```

Misurato su **quattro delle sei barre** della Figura 3, su due coppie di domini:

| barra | Figura 3 (shared/specific) | nostra misura |
|---|---|---|
| Elec | 80.55 / 19.45 | **50.10 / 49.90** |
| Cloth | 80.78 / 19.22 | **49.52 / 50.48** |
| **Movie** | 28.04 / 71.96 | **49.04 / 50.96** |
| **Book** | 71.45 / 28.55 | **47.60 / 52.40** |

Douban è il caso più severo, ed è quello che chiude la questione. Il paper mostra
Movie e Book su **lati opposti**, con 43 punti di scarto fra loro, e ci costruisce
sopra l'argomento sul "Domain Type": i domini di contenuto preferirebbero le
feature specific, quelli di consumo le shared. Noi misuriamo 49.04 e 47.60:
**1,4 punti di differenza**. Non è che i valori siano diversi — il pattern
qualitativo su cui poggia l'affermazione non esiste.

Per produrre i valori della figura servirebbe un gap di logit di 22.74 su
Elec/Cloth, −15.1 su Movie e +14.7 su Book. Misuriamo +0.06, −0.31, −0.62 e
−1.54. Su Movie il segno è giusto ma il valore è 24 volte troppo piccolo; su Book
il segno è **opposto**.

Abbiamo escluso che la figura riporti un'altra quantità: né i pesi di attention
(50.10%), né la proporzione delle norme dopo l'attention (49.52%), né quella
prima (49.43%) si avvicinano a 80/20.

Il motivo è architetturale. I logit vengono divisi per √d, e con d = 256 la scala
è 16. Differenze già piccole vengono schiacciate, e il softmax restituisce quasi
sempre metà e metà.

**Conseguenza per τ.** Scomponendo la varianza dei log-odds di τ, l'attention
spiega l'**1.3%** e la geometria degli item il **92.5%**. τ varia perché gli item
si allineano diversamente con i due sottospazi, non perché il modello decida
diversamente da utente a utente.

### 5.5 τ serve, ma come diagnostica

Sul checkpoint collassato τ vale 0.4967 con deviazione **0.006**: uguale per ogni
utente e ogni item. Sembrava un bug del nostro strumento. Non lo era: lì i due
canali sono lo stesso vettore (coseno 0.997), quindi contribuiscono per forza
allo stesso modo.

| | CDs→Instr (collassato) | Cloth→Elec (sano) |
|---|---|---|
| τ pooled | 0.4967 | 0.5031 |
| **deviazione standard** | **0.006** | **0.080** |
| τ per utente (min–max) | ~0.50 piatto | 0.310 – 0.733 |

**Il tranello:** la media è quasi identica nei due casi. Chi guarda solo la media
conclude il contrario del vero. **Il segnale è la dispersione.**

È questo che τ ha dato di utile: una spia di collasso leggibile a colpo d'occhio,
definita sulla singola raccomandazione invece che sull'intero spazio latente.

---

## 6. L'audit semantico: cosa contengono i canali (Contributo 2)

L'attribuzione dice **quanto** un canale ha pesato. Non dice **di cosa parla**.
Per una spiegazione in linguaggio naturale serve il contenuto, e qui entra
l'LLM.

Il procedimento: si raggruppano gli item vicini nel sottospazio di ciascun
dominio, un LLM (`qwen3.5:9b`) nomina i gruppi **alla cieca** — solo titoli e
categorie, mescolati, senza sapere dominio, canale o posizione — e un **secondo**
LLM (`gemma4`) verifica che quei nomi identifichino davvero i gruppi.

### 6.1 Le etichette sono valide

Test a 4 alternative su item **tenuti da parte**, mai visti durante il naming:

| | corretti | accuratezza | p |
|---|---|---|---|
| shared, source | 12/29 | 41.4% | 0.039 |
| shared, target | 17/29 | 58.6% | 0.0001 |
| base, source | 15/30 | 50.0% | 0.003 |
| base, target | 16/26 | 61.5% | 0.0001 |

Tutte significative. Lo strumento misura qualcosa, quindi il resto è
interpretabile. Era il cancello del piano: se le etichette fossero state
plausibili ma vuote, ci saremmo fermati qui.

### 6.2 Il canale shared porta una corrispondenza cross-domain, ma il confronto con base resta aperto

Il test confronta **due accoppiamenti indipendenti**: quello che il modello ha
codificato (per ogni cluster source, il cluster target col centroide più vicino)
e quello che l'LLM legge dalle sole etichette, senza vedere la geometria.

Quattro clusterizzazioni dello stesso checkpoint, granularità e seed diversi:

| run | shared | base | differenza |
|---|---|---|---|
| k=12 | 50.0% (2/4) | 36.4% | +14 |
| k=30 | 47.6% (10/21) | 32.0% | +16 |
| k=60, seed 42 | 30.2% (13/43) | 27.8% | +2 |
| k=60, seed 7 | 31.8% (14/44) | 21.7% | +10 |
| **aggregato** | **34.8%** (39/112) | **26.7%** (40/150) | **+8.2** |

Cosa si può dire:

- **il canale shared batte il caso**: 34.8% contro 25%, **p = 0.013** su 112
  coppie e quattro clusterizzazioni. Quando il modello avvicina un gruppo di
  prodotti elettronici a un gruppo di capi d'abbigliamento, quell'accostamento ha
  un senso leggibile più spesso di quanto accadrebbe a caso;
- **l'embedding grezzo no**: 26.7%, p = 0.35;
- **che shared sia meglio di base non è dimostrato**: Fisher dà **p = 0.17**.

**È comunque notevole che shared batta il caso**, perché nessuno l'ha chiesto al
modello: `cl_sim_weight` allinea i canali comuni **degli utenti**, e non esiste
alcun termine di allineamento cross-domain sugli item. La corrispondenza si è
formata da sola, propagandosi attraverso lo spazio utente condiviso.

**Una stima che si è ridimensionata.** Le prime misure davano 47.6%, ma su 21
coppie. Aggiungendo clusterizzazioni il valore si assesta intorno al 35%, e la
differenza stimata scende da 15.6 punti a 8.2. È il comportamento tipico di un
effetto misurato su pochi campioni. Per rilevare 8 punti servirebbero circa
**500 coppie per canale**, cinque volte quelle raccolte.

**Perché non basta aggiungere cluster.** Passando da 30 a 60 gruppi **entrambi**
i canali crollano: shared da 47.6% a ~31%, base da 32% a ~25%. Più cluster
significa etichette più simili fra loro dentro lo stesso dominio, che il giudice
confonde. Le coppie in più si pagano in risoluzione: è una tensione strutturale
del disegno, non una questione di quanto compute si spende.

Le due strade che resterebbero: **altri checkpoint** (repliche genuinamente
indipendenti invece di clusterizzazioni dello stesso modello) oppure una
**misura graduata** al posto della scelta forzata — chiedere al giudice un
punteggio di somiglianza sulla coppia geometrica e su una casuale, e confrontare
le distribuzioni. Stesso costo in chiamate, molta più informazione per chiamata.

### 6.3 Ma la corrispondenza è un imbuto

Quanti cluster target distinti vengono raggiunti dai 30 cluster source:

| canale | osservato (3 seed) | atteso a caso |
|---|---|---|
| **shared** | 14 (13, 14, 15) | 19.2 |
| base | 17 (16, 17, 19) | 19.2 |

Il canale base è **indistinguibile dal caso**: due grafi separati, con un solo
item in comune, non hanno motivo di corrispondersi. Il canale shared è
**significativamente più concentrato del caso**: molti cluster source convergono
su poche regioni target.

E otto dei trenta puntano su cluster target che l'LLM **non ha saputo nominare**.
Non è un effetto della dimensione: il target più attrattivo del run a 12 cluster
era il **più piccolo** dei dodici (1.297 item).

Convivono quindi due cose: **dove la corrispondenza è leggibile è anche corretta,
ma in un terzo dei casi punta su regioni che non significano nulla.**

### 6.4 Un bug che vale la pena raccontare

Il primo run dava, sul dominio source, una validità del **16.7% — sotto il caso
del 25%**. Un risultato sotto il caso è quasi sempre il segno di un errore a
monte, non di un fenomeno debole.

Lo era: leggevo gli item del source dalla decomposizione del **target**, dove
quelle righe non sono mai state addestrate. Norma media **0.021** contro 0.591.
Stavo raggruppando rumore, e infatti i cluster venivano tutti della stessa
dimensione (5116–5295), che è la firma di k-means su dati senza struttura.

Corretto l'errore, la validità del source passa a **58.3%** e le etichette
diventano distinte ("DSLR and Mirrorless Camera Accessories", "Computer Internal
Components and Peripherals") invece di dodici varianti della stessa frase.

Ora ogni dominio viene decomposto dalla propria propagazione, e lo script stampa
la norma media dei canali: se ricompare un valore intorno a 0.02, il problema si
vede prima di diventare un risultato.

---

## 7. Un risultato ritirato

All'inizio sembrava emergere che più un utente ha storia nel dominio target, più
la raccomandazione dipende dal transfer. Sarebbe stato il contrario
dell'assunzione standard sul cross-domain.

Con quattro misure invece di due, l'effetto **cambia segno**:

| run | τ (storia 5–19) | τ (storia 20+) | differenza |
|---|---|---|---|
| Elec→Cloth, org 0.1 | 0.4145 | 0.4813 | **+0.067** |
| Cloth→Elec, org 1 | 0.5010 | 0.5231 | **+0.022** |
| Cloth→Elec, org 0.01 | 0.5000 | 0.4866 | **−0.013** |
| Cloth→Elec, org 10 | 0.4758 | 0.4592 | **−0.017** |

Cambia segno variando un solo peso di loss, sugli stessi dati e sugli stessi
utenti. Non è un effetto debole: è un artefatto degli iperparametri. Ritirato.

---

## 8. Limiti

- **Un seed per configurazione.** Niente è stato replicato.
- I confronti tra le due direzioni cambiano più variabili insieme. Lo sweep
  interno a Cloth→Elec è a variabile singola, ed è quello su cui poggia §5.1.
- I dataset sono 10-core: **nessun utente davvero cold-start**.
- τ descrive **come è composto** il punteggio, non cosa succederebbe azzerando un
  canale e ricalcolando la classifica.
- τ alto **non** significa raccomandazione migliore. È un'attribuzione, non una
  valutazione.
- La decomposizione esatta richiede `fuse_mode='attention'`, e solo gli utenti
  sovrapposti hanno un τ definito.
- I probe sono **a soffitto** (98–100%): lì non distinguono più configurazioni
  diverse, e il confronto va appoggiato sul gap di dCor.
- Il gap di dCor dipende dal numero di utenti campionati. Confrontabile solo a
  `--dcor_sample` uguale.
- **Sull'attention restano non misurate le due barre di Sport&Cloth**, quattro
  su sei sono verificate su due coppie indipendenti.
- **Bug noto in `dgcdr.py`**, rilevante per il Contributo 2: il caricamento degli
  embedding testuali indicizza l'array del dominio source con l'ID fuso, che è
  compattato. Su CDs/Instruments mappa correttamente solo i 3.609 item del
  target, ne mappa 23 sul prodotto sbagliato, e lascia a zero i 3.609 del source.
  Nessuno dei run qui usava `use_text_embeddings=True`.
- **Una previsione sbagliata**, annotata: pensavo che l'informazione di dominio
  stesse nelle componenti principali della base. Rimosse le prime cinquanta su
  256, il probe MLP resta al 98.6% (controllo con direzioni casuali: 98.3%).
- **Una seconda previsione sbagliata**: mi aspettavo che l'audit semantico (§6)
  mostrasse che il canale shared non aggiunge nulla rispetto all'embedding
  grezzo. È emerso il contrario. Vale la pena registrarlo perché era una
  previsione motivata — non esiste una loss di allineamento sugli item — e si è
  rivelata falsa.
- L'audit semantico poggia su **quattro clusterizzazioni di un solo
  checkpoint**. Repliche su modelli diversi non ci sono: il tentativo è fallito
  perché i checkpoint allenati sulla VM riferiscono directory di dataset con
  nomi diversi da quelli locali.
- **Il test di corrispondenza si degrada al crescere dei cluster**, perché le
  etichette diventano più simili fra loro e il giudice le confonde. Non è quindi
  possibile comprare potenza statistica semplicemente alzando la granularità.
- **Una stima ridimensionata**: la corrispondenza del canale shared era stata
  riportata al 47.6% su 21 coppie; su 112 coppie scende al 34.8%. Le prime
  misure erano ottimistiche.

---

## 9. Prossimi passi

**1. Sport&Cloth**, le ultime due barre della Figura 3 rimaste non misurate.

**2. Replica con più seed**, per stabilire la stabilità delle misure.

**3. Controfattuale**: azzerare il canale shared, ricalcolare le classifiche, e
passare dall'attribuzione alla causalità.

**4. Chiudere il confronto shared contro base** (§6.2). Aggiungere cluster non
funziona: degrada la misura. Le due strade praticabili sono ripetere l'audit su
**altri checkpoint** — da fare sulla VM, dove quei modelli si caricano — oppure
sostituire la scelta forzata con una **misura graduata**, che estrae molta più
informazione da ogni chiamata.

---

## 10. Il contributo

L'obiettivo è **spiegare le raccomandazioni cross-domain in modo verificabile**.

**Il contributo metodologico** è il framework di attribuzione. Tre proprietà lo
distinguono dagli explainer esistenti:

1. **è esatto per costruzione** — non stima e non approssima, rifà l'aritmetica
   del modello;
2. **si autoverifica a ogni esecuzione** — ricompone i canali e li confronta con
   l'output vero, con errore relativo tra 1e-07 e 3e-07;
3. **si rifiuta di produrre spiegazioni** quando non può garantirle, invece di
   restituire un'approssimazione plausibile.

Nessun explainer basato su LLM offre queste garanzie: genera testo coerente con
la cronologia dell'utente, senza alcun legame dimostrabile con il calcolo del
modello.

**Da qui nasce τ**, il transfer ratio: una misura per singola raccomandazione,
non per l'intero spazio latente come le metriche di disentanglement esistenti.

**La validazione è la sezione 5.** La domanda che un lettore pone a un explainer
è "a cosa serve". La risposta è che, applicato a un modello che riproduce i
risultati pubblicati entro il 3%, ha reso visibili tre proprietà che nessuna
metrica esistente mostrava:

1. l'ortogonalità non compra indipendenza, e non lo farebbe nemmeno con pesi
   10²³ volte maggiori;
2. sul lato item il disentanglement non avviene affatto — invisibile alla
   valutazione esistente, che misura solo i canali utente;
3. l'attention è quasi inerte, e la figura del paper che la descrive non è
   riproducibile.

Queste scoperte non sono un lavoro parallelo: sono **il motivo per cui serve uno
strumento del genere**. Sono anche il motivo per cui una spiegazione ingenua
sarebbe fuorviante — dire a un utente "ti consigliamo questo per i gusti
trasferiti dall'altro dominio" presuppone che quel canale contenga davvero
conoscenza trasferita, e la sezione 5 mostra che non è così.

**Il Contributo 2 aggiunge il livello semantico** (§6). L'attribuzione dice
quanto pesa un canale; l'audit dice di cosa parla. Ed è la parte che ha prodotto
il primo risultato **positivo** sul modello: il canale shared porta una
corrispondenza cross-domain che un LLM riconosce (34.8% contro un caso del 25%,
p = 0.013), mentre l'embedding grezzo resta al caso — pur non esistendo alcuna
loss che allinei gli item fra i due domini. Che shared sia meglio di base resta
però non dimostrato (p = 0.17).

**Il limite da dichiarare.** τ non ha ancora prodotto un risultato stabile sui
pattern di transfer: l'unico candidato è stato ritirato (§7). Il suo valore
dimostrato è diagnostico — riconoscere il collasso della separazione — e come
base per l'audit semantico.
