# Contributo 2 — Audit semantico dei sottospazi disentangled

Piano rivisto dopo la Fase 1. Branch `explainability-2`, 26 luglio 2026.

La prima versione di questo piano è stata in gran parte falsificata dai suoi
stessi cancelli. Quello che segue è la versione che resta in piedi.

---

## 1. Cosa ha stabilito la Fase 1

La Fase 1 doveva accertare che i sottospazi avessero struttura semantica prima
di coinvolgere un LLM. Ha risposto, e ha risposto più di quanto le fosse stato
chiesto.

### I due canali item sono la stessa struttura

Correlazione tra le matrici di similarità item-item indotte dai due canali,
misura deterministica con null calibrato:

| | coseno(e^c, e^s) | corr. delle strutture | null |
|---|---|---|---|
| Elec→Cloth (sano) | 0.0365 | **+0.9984** | +0.0002 |
| CDs→Instr (collassato) | 0.9973 | +0.9997 | +0.0002 |

Su questa misura il modello sano è indistinguibile da quello collassato.

### La causa: i gate degli item hanno imparato la stessa cosa

| gate | cos(gate_c, gate_s) | sovrapposizione | medie |
|---|---|---|---|
| utente | 0.0505 | 4.7% | 0.506 / 0.440 |
| **item** | **0.9985** | **98.2%** | 0.849 / 0.852 |

Il disentanglement degli item **non avviene**, nemmeno nel modello che
riproduce il paper. Sul lato utente i gate selezionano coordinate quasi
disgiunte; sul lato item sono la stessa funzione.

Questo non era mai emerso perché `evaluate_disentanglement.py` misura solo i
canali utente. È un buco della valutazione esistente, ed è esso stesso un
risultato da riportare.

### L'informazione di dominio è irremovibile per via lineare

Rimuovendo iterativamente le direzioni più discriminative (INLP) dal canale
shared degli utenti:

| iterazioni | probe lineare | probe MLP |
|---|---|---|
| 0 | 98.82% | 99.63% |
| 5 | **54.64%** | 98.16% |
| 20 | **50.07%** | 98.28% |
| 40 | 48.50% | 96.73% |

Cinque direzioni contengono tutta l'informazione linearmente decodificabile.
Toglierle porta un probe lineare al caso esatto e lascia un probe non lineare
al 98%.

Ne segue perché ogni intervento provato finora ha fallito: l'ortogonalità è un
vincolo lineare, l'allineamento è un coseno, il discriminatore avversariale era
una rete piccola. **Sono tutti strumenti lineari o quasi, applicati a
informazione che non è linearmente accessibile.**

E ne segue un avvertimento metodologico: chi valuta il disentanglement con un
probe lineare, dopo un intervento del genere, legge 50% e conclude di aver
avuto successo.

### Una previsione sbagliata, annotata

Avevo previsto che proiettando via le componenti principali della base il probe
sarebbe crollato. Rimosse le prime 50 su 256, resta al 98.6% (controllo con 50
direzioni casuali: 98.3%). L'informazione non sta nelle direzioni ad alta
varianza: è distribuita, e la ridondanza è più forte di quanto ipotizzassi.

Il rango effettivo degli embedding è ~30 su 256 sia per utenti sia per item,
con il 54–57% della varianza nelle prime dieci componenti. La struttura è
fortemente a basso rango, ma l'identità di dominio non vive lì.

---

## 2. Cosa muore e cosa resta del piano originale

**M2 — confronto tra concetti shared e specific: eliminata.** Presupponeva due
canali distinti da confrontare. Sugli item ce n'è uno solo: l'LLM produrrebbe
due liste identiche e la misura darebbe zero per costruzione, senza che quello
zero significhi niente.

**M0 — coerenza dei cluster: assorbita e superata.** La correlazione tra
strutture è più informativa della purezza dei cluster, è deterministica e non
dipende da k-means. Resta come controllo secondario.

**M1 — validità delle etichette: resta, indispensabile.** Senza, il resto è
"abbiamo chiesto a un LLM e ha detto cose sensate".

**M3 — corrispondenza cross-domain: promossa a cuore del contributo.** Confronta
i due *domini*, non i due canali, quindi il collasso dei gate non la tocca. È
anche l'unica domanda rimasta a cui solo un LLM può rispondere.

---

## 3. Il contributo ridefinito

> `cl_sim_weight` esiste per allineare i canali comuni dei due domini. È l'unico
> meccanismo di DGCDR che punta all'invarianza di dominio. **Compra un
> allineamento semantico reale?**

I concetti estratti dal canale shared del dominio source devono corrispondere a
quelli del target: "rock anni '70" da un lato deve trovare "chitarre elettriche
e amplificatori valvolari" dall'altro. Se non corrispondono, l'allineamento è
una coincidenza geometrica senza contenuto.

È la domanda naturale dopo aver stabilito che l'ortogonalità non separa nulla:
l'altra metà del meccanismo funziona?

---

## 4. Le misure

### M1 — Validità delle etichette *(cancello)*

Si tengono da parte alcuni item del cluster. Si mostra all'LLM l'etichetta e *N*
insiemi di item, uno vero e gli altri da cluster diversi, e gli si chiede quale
corrisponde. Accuratezza sopra 1/N significa che l'etichetta cattura davvero il
cluster.

Se fallisce, lo strumento non misura e M3 non è interpretabile.

### M3 — Corrispondenza cross-domain *(risultato principale)*

1. Estrarre i concetti dagli item **source** usando il canale shared del source.
2. Estrarre i concetti dagli item **target** usando il canale shared del target.
3. Nominarli separatamente e alla cieca (mai il nome del dominio nel prompt).
4. Chiedere a un secondo LLM di accoppiarli.
5. Confrontare con due riferimenti.

| riferimento | cosa dice |
|---|---|
| accoppiamento casuale | il null: quanto si ottiene per caso |
| stessa misura sul canale **base** | quanto si otterrebbe **senza** disentanglement |

Il secondo è il controllo che conta. Il canale base è l'embedding GNN grezzo,
senza alcuna separazione: se i concetti shared non si accoppiano meglio di
quelli base, **la loss di allineamento non ha aggiunto nulla di semantico**.

### M4 — Mappa concettuale descrittiva *(secondario)*

Poiché sugli item esiste una sola struttura, ha comunque senso dire *cosa*
codifica. Non è un test, è una descrizione — ma è la risposta leggibile alla
domanda "cosa rappresenta davvero il modello", e serve al lettore del paper.

---

## 5. Prerequisiti

| | stato |
|---|---|
| Checkpoint sano con `item_disentangle=True` | ✅ Elec→Cloth locale |
| Metadati Elec + Cloth | ✅ in cache, 135.217 item, risoluzione 100% |
| Risoluzione ID → token | ✅ verificata |
| Estrazione dei concetti | ✅ `concepts.py` |
| Endpoint LLM | ⚠️ serve, `verbalize.py` parla con qualunque endpoint OpenAI-compatible |

Il vincolo dei metadati, che nella prima versione del piano bloccava tutto, è
risolto: il caricamento filtrato in streaming impiega 38 secondi sui 23 GB di
dump.

---

## 6. Implementazione

`concepts.py` esiste già. Restano:

**`concept_naming.py`** — etichettatura cieca. Il prompt riceve solo titoli e
categorie, mescolati; mai ID, nome del dominio, nome del canale o posizione nel
ranking. Deve poter rispondere "nessun tema comune": quella risposta è un dato.

**`concept_eval.py`** — `label_validity` (M1) e `cross_domain_matching` (M3),
ciascuna con il proprio null.

**`audit_concepts.py`** — CLI sul modello di `explain_dgcdr.py`. Deve rifiutarsi
di girare su un modello collassato, e ora sappiamo che il controllo va fatto
**per canale**: sugli item di questo checkpoint la correlazione tra strutture è
0.9984, quindi lo script deve dirlo invece di procedere in silenzio.

---

## 7. Fasi

**Fase 2 — naming e M1.** Cancello: se le etichette non superano il test *N*-way,
ci si ferma e si rivede lo strumento.

**Fase 3 — M3.** Il risultato, con entrambi i riferimenti (casuale e canale base).

**Fase 4 — M4 e replica.** Mappa descrittiva, due LLM diversi, e se possibile una
seconda coppia di domini.

---

## 8. Come può fallire

- **Le etichette non sono valide** (M1) → lo strumento non misura, ci si ferma.
- **I concetti shared si accoppiano come quelli base** → l'allineamento non
  aggiunge nulla. È il risultato che mi aspetto, ed è pubblicabile.
- **Nessuno dei due si accoppia sopra il caso** → i sottospazi non hanno
  corrispondenza cross-domain di alcun tipo; risultato più forte, ma va escluso
  che dipenda da cluster incoerenti (per questo M1 viene prima).
- **Circolarità** → l'LLM che nomina e quello che accoppia devono essere separati
  e ciechi.
- **Contaminazione da popolarità** → i cluster potrebbero riflettere la
  popolarità invece del contenuto. Da controllare correlando dimensione dei
  cluster e popolarità media.

---

## 9. Nota onesta sul ruolo dell'LLM

I risultati più forti raccolti finora — collasso dei gate item, struttura
identica dei due canali, irremovibilità lineare dell'informazione di dominio —
**non hanno richiesto alcun LLM**. Sono misure di algebra lineare con il
proprio null.

Il Contributo 2 resta l'unico punto in cui un LLM fa qualcosa che nessun altro
strumento può fare: leggere il contenuto dei prodotti e giudicare se due insiemi
di concetti, estratti da domini diversi, parlino della stessa cosa. Vale la pena
tenerlo, ma con l'aspettativa giusta: è una misura in più, non la spina dorsale
del lavoro.
