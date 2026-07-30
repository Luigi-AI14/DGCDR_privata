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

**Il meccanismo: il modello imbroglia rimpicciolendo.** La penalità è sul
prodotto scalare **grezzo**, e il prodotto scalare si scompone così:

```
e^c · e^s  =  ‖e^c‖ · ‖e^s‖ · cos(angolo)
```

Ci sono quindi due modi di ridurlo: ruotare i vettori — che è l'obiettivo — o
**accorciarli** — che non serve a niente. Misurando le norme lungo uno sweep su
Elec→Cloth:

| `cl_org_weight` | ‖e^c‖ | ‖e^s‖ | coseno |
|---|---|---|---|
| 0.01 | 1.146 | 1.423 | 0.159 |
| 0.1 | 1.218 | 1.393 | 0.037 |
| **10** | **0.286** | 1.212 | 0.014 |

Da 0.1 a 10 il canale shared **si accorcia di 4,3 volte**, mentre lo specific
resta dov'era. Scomponendo la riduzione del prodotto scalare: **il 57% viene
dall'accorciamento, solo il 38% dalla rotazione.**

E accorciare non toglie informazione: un vettore diviso per quattro contiene
esattamente quello che conteneva prima. Il probe infatti **standardizza le
feature**, quindi la scala gli è invisibile — ed è la ragione per cui legge 99%
comunque, mentre coseno e loss migliorano.

Il modello ha speso la maggior parte dello sforzo su una riduzione di scala che
la loss premia e che sull'obiettivo non incide.

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

### 5.6 Il controfattuale: τ predice cosa si rompe, ma il canale non serve

Finora τ diceva **come è composto** un punteggio. Non diceva cosa succederebbe
togliendo un canale — sono due domande diverse, e solo la seconda è causale.

Il test: azzerare un canale, ricalcolare le classifiche, e guardare cosa cambia.

**τ è causalmente valida, e replica su due dataset.** Le raccomandazioni che
escono dalla top-20 quando si azzera il canale shared sono quelle che τ aveva
indicato come shared-driven:

| τ medio | Elec→Cloth | Douban Movie→Book |
|---|---|---|
| raccomandazioni **cadute** | 0.4833 (n=461) | 0.4542 (n=501) |
| raccomandazioni **rimaste** | 0.4160 (n=5539) | 0.3748 (n=5499) |
| differenza | **+0.0673** | **+0.0794** |
| p (permutazione) | **< 0.0001** | **< 0.0001** |
| corr(τ, nuovo rango) | +0.097 | +0.118 |

τ non è contabilità che torna: **predice cosa si rompe**, su due coppie di
domini molto diverse. È il test più severo che si possa fare a questo
contributo, e chiude il limite che era dichiarato in §7.

L'effetto è modesto — 7-8 punti di scarto, correlazione intorno a +0.1 —
significativo grazie ai 6.000 campioni per dataset, non grande.

**Ma il canale non serve all'accuratezza.**

| canale azzerato | Elec→Cloth | Douban |
|---|---|---|
| intatto | 0.0202 | 0.0998 |
| **shared** | 0.0210 (**+4.0%**) | 0.1036 (**+3.7%**) |
| specific | 0.0194 (−4.0%) | 0.1042 (+4.3%) |
| base | 0.0194 (−4.0%) | 0.0992 (−0.6%) |

Su entrambi i dataset, **togliere il canale shared non costa nulla**: semmai
migliora di circa il 4%. E oltre il 90% della top-20 resta identica.

**Vale su tutto lo sweep**, quindi non dipende dalla taratura
dell'ortogonalità. Elec→Cloth, variando solo `cl_org_weight`:

| `cl_org_weight` | 0.01 | 0.1 | 10 |
|---|---|---|---|
| costo di azzerare shared | +4.2% | +4.0% | +3.6% |
| top-20 che sopravvive | 93.2% | 92.3% | **99.3%** |
| τ delle cadute | 0.4762 | 0.4833 | **0.0491** |
| τ delle rimaste | 0.4051 | 0.4160 | **0.0338** |
| differenza (tutte p ≤ 0.0001) | +0.0711 | +0.0673 | +0.0153 |

τ predice cosa si rompe a **tutti e tre i pesi**. Ma a `org=10` i valori di τ
crollano di dieci volte e il 99,3% della top-20 sopravvive: è la firma
dell'accorciamento descritto in §5.1. Spingendo sull'ortogonalità il modello non
rende il canale indipendente, lo **spegne**.

**Una correzione rispetto alla prima stesura.** Avevo scritto che i controlli
escludevano un artefatto, perché su Elec→Cloth azzerare `specific` o `base`
costava il 4%. Su Douban non è così: azzerare `specific` migliora del 4,3% e
`base` è neutro. L'asimmetria fra shared e specific **non replica**, quindi non
si può sostenere.

Quello che replica, e che resta affermabile, è più semplice e più radicale:
**nessuna ablazione sposta l'accuratezza oltre il 4%, in nessuna direzione, su
nessuno dei due dataset.** Il modello è ridondante al punto che rimuovere un
terzo della sua rappresentazione quasi non si nota.

*(0.0202 contro lo 0.0253 del paper è una differenza di protocollo: 300 utenti
campionati e mascheramento nostro, contro la valutazione completa di RecBole. Il
confronto intatto/ablato è interno e coerente.)*

**Come vanno letti insieme.** Le due affermazioni sono compatibili:

> **τ misura correttamente una quantità che nel modello conta poco.**

Lo strumento fa quello che promette. Il canale che misura è quasi decorativo dal
punto di vista dell'accuratezza — in linea con §5.1, dove moltiplicare
`cl_org_weight` per mille non spostava Recall di un decimale.

---

## 6. Una strada esplorata e abbandonata: l'audit semantico

Abbiamo provato a usare un LLM come strumento di misura, per dare un nome
leggibile a cosa contengono i canali. Il procedimento: raggruppare gli item
vicini in ciascun sottospazio, farli nominare alla cieca a un LLM che vede solo
titoli e categorie, e verificare con un secondo modello che quei nomi
identifichino davvero i gruppi.

**Lo strumento funzionava.** Le etichette superavano un test di validità severo
— dal 41% al 62% di riconoscimenti contro un caso del 25%, tutti significativi —
quindi si può davvero tradurre un sottospazio in una frase leggibile. Ne
uscivano concetti sensati come *"DSLR and Mirrorless Camera Accessories"* o
*"Computer Internal Components"*.

**La domanda scientifica però ha dato risposta negativa.** Volevamo sapere se i
concetti del canale shared dei due domini si corrispondessero — se cioè
`cl_sim_weight` producesse un allineamento semantico reale. Il risultato è
sembrato esserci all'inizio (47.6% su 21 coppie) e si è dissolto man mano che
arrivavano dati: 34.0% su 159 coppie, con la differenza rispetto all'embedding
grezzo scesa da 15.6 punti a 4.6. Una misura appaiata costruita apposta per
avere più potenza ha dato p = 0.148.

Era peraltro prevedibile dall'architettura, e l'avevamo annotato prima di
misurare: `cl_sim_weight` allinea i canali comuni **degli utenti**, e non esiste
alcun termine che allinei gli item fra i due domini.

Il codice è stato rimosso dal repository perché non regge un contributo. Resta
qui la nota, perché un vicolo cieco documentato vale più di uno cancellato: se
la domanda "avete provato a guardare dentro i canali con un LLM?" dovesse
tornare, la risposta è sì, ed è questa.

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
- Il controfattuale (§5.6) è stato misurato su due checkpoint, 300 utenti
  campionati ciascuno. La validazione di τ replica; il comportamento dei canali
  di controllo no.
- **Un'affermazione ritirata**: avevo scritto che azzerare `specific` o `base`
  costa il 4% mentre azzerare `shared` no, presentandolo come controllo. Vale
  su Elec→Cloth ma non su Douban, dove anche `specific` migliora. L'asimmetria
  fra i canali non è sostenibile.
- τ alto **non** significa raccomandazione migliore. È un'attribuzione, non una
  valutazione.
- La decomposizione esatta richiede `fuse_mode='attention'`, e solo gli utenti
  sovrapposti hanno un τ definito.
- I probe sono **a soffitto** (98–100%): lì non distinguono più configurazioni
  diverse, e il confronto va appoggiato sul gap di dCor.
- Il gap di dCor dipende dal numero di utenti campionati. Confrontabile solo a
  `--dcor_sample` uguale.
- **Sull'attention restano non misurate le due barre di Sport&Cloth**: il
  dataset (149k utenti, 149k item) eccede i 12 GB di VRAM disponibili. Quattro
  barre su sei sono verificate, su due coppie di domini indipendenti.
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
- **Un risultato ritirato** (§6): la corrispondenza cross-domain del canale
  shared era stata riportata come positiva, ed è svanita all'aumentare dei dati.

---

## 9. Prossimi passi

**1. Sport&Cloth su hardware più capiente**, per completare la Figura 3. In
locale non è possibile: il training da solo richiede ~11,7 GB di VRAM.

**2. Replica con più seed**, per stabilire la stabilità delle misure.

**3. La loss di ortogonalità normalizzata.** È l'esperimento che nasce
direttamente da §5.1: penalizzare il coseno invece del prodotto scalare grezzo.
Il coseno è invariante di scala, quindi la scorciatoia dell'accorciamento — che
oggi spiega il 57% della riduzione della loss — sparisce, e al modello resta
solo la rotazione.

È una previsione falsificabile. Se il probe scende, la loss era formulata male e
la correzione è di una riga. Se resta al 99%, il difetto è concettuale e non
implementativo, e nemmeno l'ortogonalità vera basta. Entrambi gli esiti sono
risultati, ed è anche l'unico punto in cui il lavoro indica una correzione
invece di limitarsi a constatare un fallimento.

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

**Una strada è stata esplorata e abbandonata** (§6): usare un LLM per dare un
nome ai sottospazi. Lo strumento funzionava, ma la domanda scientifica ha dato
risposta negativa e il codice è stato rimosso.

**La validazione causale** (§5.6) è la prova più severa che il framework abbia
superato: le raccomandazioni che crollano azzerando il canale shared sono quelle
che τ aveva indicato, con p < 0.0001. τ non è una scomposizione che torna per
costruzione, misura qualcosa che il modello fa davvero.

**Il limite da dichiarare.** τ non ha prodotto un risultato stabile sui pattern
di transfer: l'unico candidato è stato ritirato (§7). E la quantità che misura
correttamente conta poco per l'accuratezza — azzerare il canale shared non
peggiora le raccomandazioni. Lo strumento è valido; l'oggetto che misura è
meno importante di quanto l'architettura suggerisca.
