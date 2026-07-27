# Cosa abbiamo fatto e cosa abbiamo trovato

Documento di lavoro, 26 luglio 2026. Branch `explainability-1`.
Cinque checkpoint analizzati: CDs→Instruments, Elec→Cloth, e Cloth→Elec a tre
valori di `cl_org_weight`.

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
Se non coincidono, si ferma. Sui cinque checkpoint l'errore relativo sta tra
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

Scarti tra l'1% e il 3%, cioè normale variabilità da seed. **Tutto quello che
segue riguarda il modello pubblicato**, non una nostra versione difettosa.

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

| Elec&Cloth | Figura 3 | nostra misura |
|---|---|---|
| shared | 80.55% / 80.78% | **50.10% / 49.52%** |
| specific | 19.45% / 19.22% | 49.90% / 50.48% |

Per produrre 80.55/19.45 servirebbe un gap di logit pari a 22.74. Nel nostro
modello è +0.06 sul source e −0.31 sul target.

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

## 6. Un risultato ritirato

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

## 7. Limiti

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
- **Sull'attention abbiamo misurato una coppia di domini su tre.** Per ora si può
  dire "su Elec/Cloth non è riproducibile", non "la Figura 3 è sbagliata".
- **Bug noto in `dgcdr.py`**, rilevante per il Contributo 2: il caricamento degli
  embedding testuali indicizza l'array del dominio source con l'ID fuso, che è
  compattato. Su CDs/Instruments mappa correttamente solo i 3.609 item del
  target, ne mappa 23 sul prodotto sbagliato, e lascia a zero i 3.609 del source.
  Nessuno dei run qui usava `use_text_embeddings=True`.
- **Una previsione sbagliata**, annotata: pensavo che l'informazione di dominio
  stesse nelle componenti principali della base. Rimosse le prime cinquanta su
  256, il probe MLP resta al 98.6% (controllo con direzioni casuali: 98.3%).

---

## 8. Prossimi passi

**1. Douban Movie↔Book.** Il run con il miglior rapporto valore/costo: testa il
punto più estremo della Figura 3 (28.04/71.96) e aggiunge una terza coppia a
tutto il resto.

**2. Replica con più seed**, per stabilire la stabilità delle misure.

**3. Controfattuale**: azzerare il canale shared, ricalcolare le classifiche, e
passare dall'attribuzione alla causalità.

**4. Contributo 2**, descritto nel piano a parte.

---

## 9. Il contributo

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

**Il limite da dichiarare.** τ non ha ancora prodotto un risultato scientifico
stabile sui pattern di transfer: l'unico candidato è stato ritirato (sezione 6).
Al momento il suo valore dimostrato è diagnostico — riconoscere il collasso della
separazione — e come base per il Contributo 2, che aggiunge il livello semantico
alle spiegazioni numeriche.
