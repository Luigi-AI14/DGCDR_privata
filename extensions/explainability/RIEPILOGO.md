# Riepilogo del lavoro — per riprendere da capo

Aggiornato al 31 luglio 2026. Contiene solo ciò che ha validità accertata.
Il dettaglio completo con tutti i numeri è in `RISULTATI.md`.

---

## 1. Dove sta tutto

**Branch da usare: `normalized-losses`.** Contiene tutto il lavoro. Gli altri
sono stadi precedenti e non servono più:

| branch | contenuto |
|---|---|
| **`normalized-losses`** | tutto, incluse le tre varianti di formulazione |
| `explainability-2` | tutto tranne le varianti |
| `explainability-1` | fermo a prima del controfattuale, obsoleto |
| `explainability` | lavoro precedente, non nostro |

**Ambiente:** conda `dgcdr_env`, Python 3.8, `recbole==1.0.1`, torch 2.4.1+cu118.
L'interprete è `C:/Users/Utente/miniconda3/envs/dgcdr_env/python.exe` — `python`
non è nel PATH della shell.

**Per l'LLM:** Ollama in locale con `qwen3.5:9b` e `gemma4:latest`. Va avviato
(`ollama serve`) perché non parte da solo.

---

## 2. Il framework

Decompone il punteggio di una raccomandazione nei canali che il modello usa:

```
punteggio = contributo shared + contributo specific + contributo collaborativo grezzo
```

La decomposizione è **esatta**, non approssimata: DGCDR fonde i canali con una
somma e il punteggio è un prodotto scalare, quindi si applica la proprietà
distributiva. Ogni esecuzione ricompone i pezzi e li confronta con l'output vero
del modello, e si ferma se non tornano — sui nove checkpoint l'errore relativo
sta fra 1e-07 e 3e-07.

Da qui nasce **τ (transfer ratio)**, fra 0 e 1: quanto una singola
raccomandazione dipende dal canale shared invece che da quello specific.

### I file

| file | cosa fa |
|---|---|
| `channels.py` | riapre gli embedding nei canali, verifica la ricostruzione |
| `attribution.py` | matrice dei contributi, τ, rilevanza |
| `counterfactual.py` | azzera un canale e ricalcola le classifiche |
| `fidelity.py` | misura se il testo generato trasporta il numero |
| `metadata.py` | ID → titoli e categorie, con cache filtrata |
| `verbalize.py` | prompt vincolato e chiamata all'LLM |
| `test_decomposition.py` | autotest, 8 controlli |
| `../../explain_dgcdr.py` | script principale |
| `../../run_counterfactual.py` | ablazione dei canali |
| `../../evaluate_disentanglement.py` | metriche di disentanglement, corrette |

### I comandi

```bash
python -m extensions.explainability.test_decomposition -m saved/<ckpt>.pth
python explain_dgcdr.py -m saved/<ckpt>.pth --num_users 500 --topk 10
python explain_dgcdr.py -m saved/<ckpt>.pth --metadata_cache item_metadata/cache_elec_cloth.json --llm_model qwen3.5:9b
python run_counterfactual.py -m saved/<ckpt>.pth --num_users 300
python evaluate_disentanglement.py -m saved/<ckpt>.pth --dcor_sample 5000
```

Vincoli: serve `fuse_mode='attention'` (con `concat` la decomposizione non
sarebbe esatta e lo script si rifiuta di girare), e solo gli utenti sovrapposti
hanno un τ definito.

---

## 3. Cosa abbiamo trovato

### 3.1 I nostri modelli riproducono il paper

Confronto con la Tabella 3, su tre direzioni e due dataset:

| | Recall@20 paper | nostro | scarto |
|---|---|---|---|
| Elec→Cloth | .0260 | .0253 | −2.7% |
| Cloth→Elec | .0403 | .0397 | −1.5% |
| Movie→Book (Douban) | .1369 | .1341 | −2.0% |

**È il punto che regge tutto il resto**: le critiche riguardano il modello
pubblicato, non una nostra versione difettosa.

### 3.2 Lo strumento di misura esistente era difettoso

`evaluate_disentanglement.py` aveva tre difetti, tutti nella direzione di far
sembrare il modello migliore:

1. probe su feature non standardizzate → sottostima di **17 punti** (81.79% → 98.82%);
2. dCor confrontato con uno zero irraggiungibile: l'estimatore è distorto verso
   l'alto, due gaussiane indipendenti a N=5000, D=256 danno 0.41. Ora c'è il null
   per permutazione;
3. 19 GB di memoria per la matrice delle distanze. Ora sotto-campiona.

Con il difetto 1, Elec→Cloth sembrava avere un disentanglement score del 18%.
Il valore vero è **1.18%**.

### 3.3 L'ortogonalità non compra indipendenza

Sweep su Cloth→Elec, variando **solo** `cl_org_weight`:

| | 0.01 | 1 | 10 |
|---|---|---|---|
| coseno | 0.2164 | 0.0119 | 0.0068 |
| dCor gap | +0.7305 | +0.6726 | +0.6479 |
| probe MLP su e^c | 99.92% | 99.77% | 99.91% |
| Recall@20 | .0395 | .0397 | .0401 |

Peso ×1000 → canali 32 volte più ortogonali, gap di dCor −11%, probe fermo.
Estrapolando, servirebbe un peso dell'ordine di **10²³** per azzerare il gap.

**Il meccanismo:** la penalità è sul prodotto scalare grezzo, che si scompone in
‖e^c‖ · ‖e^s‖ · cos. Il modello lo riduce **accorciando** invece che ruotando —
a `org=10` la norma di e^c crolla da 1.218 a 0.286 mentre e^s resta a 1.2. Il
57% della riduzione viene dall'accorciamento, il 38% dalla rotazione. E
accorciare non toglie informazione: il probe standardizza le feature, quindi la
scala gli è invisibile.

### 3.4 Le correzioni ovvie non funzionano

Abbiamo riscritto la loss in forma normalizzata (coseno invece del prodotto
scalare) e tolto la divisione per √d dall'attention, separatamente e insieme:

| | pubblicata | loss norm. | senza √d | entrambe |
|---|---|---|---|---|
| attention shared | 49.5% | ~50% | 74.7% | 62.5% |
| ‖e^c‖ | 1.218 | 1.231 | 1.442 | 1.409 |
| dCor gap | +0.649 | +0.685 | +0.750 | +0.724 |
| **probe MLP** | 99.13% | 99.77% | 99.82% | 99.86% |
| Recall@20 | .0253 | .0251 | .0252 | .0252 |

Entrambe fanno ciò per cui erano pensate — la norma non crolla più, l'attention
torna selettiva — e **nessuna tocca la fuga di informazione**. Il probe resta
fra 99.1% e 99.9% in tutte e quattro le formulazioni.

**Conclusione:** il problema non è come i vincoli sono scritti, è cosa un
vincolo geometrico può fare. Non stiamo dicendo che DGCDR ha sbagliato una
formula, ma che quella famiglia di vincoli non può ottenere ciò che promette.

### 3.5 L'informazione di dominio è irremovibile per via lineare

Rimuovendo iterativamente le direzioni più discriminative (INLP) dal canale
shared:

| direzioni rimosse | probe lineare | probe MLP |
|---|---|---|
| 0 | 98.82% | 99.63% |
| 5 | **54.64%** | 98.16% |
| 20 | **50.07%** | 98.28% |
| 40 | 48.50% | 96.73% |

Cinque direzioni contengono tutta l'informazione linearmente decodificabile.
Toglierle porta un probe lineare al caso esatto e lascia quello non lineare al
98%. È la spiegazione di §3.4: gli strumenti provati sono tutti lineari o quasi.

**Avvertimento metodologico:** chi valuta con un probe lineare, dopo un
intervento del genere, legge 50% e dichiara di aver risolto.

### 3.6 Sul lato item la separazione non avviene affatto

I due canali si ottengono filtrando lo stesso vettore con due gate appresi:

| gate | cos(gate_c, gate_s) | sovrapposizione |
|---|---|---|
| utente | 0.0505 | 4.7% |
| **item** | **0.9985** | **98.2%** |

Sul lato item i due gate hanno imparato la stessa funzione, e i due canali
inducono la stessa geometria: la correlazione fra le matrici di similarità
item-item è **+0.9984** (null +0.0002). Su questa misura il modello sano è
indistinguibile da uno collassato.

Non era mai emerso perché `evaluate_disentanglement.py` misura **solo i canali
utente**.

### 3.7 L'attention è quasi inerte, e la Figura 3 si spiega

| barra | Figura 3 | nostra misura | senza √d |
|---|---|---|---|
| Elec | 80.55 / 19.45 | 50.10 / 49.90 | **74.6 / 25.4** |
| Cloth | 80.78 / 19.22 | 49.52 / 50.48 | **74.7 / 25.3** |
| Movie | 28.04 / 71.96 | 49.04 / 50.96 | non misurato |
| Book | 71.45 / 28.55 | 47.60 / 52.40 | non misurato |

Con il codice pubblicato l'attention sta a 50/50 su quattro barre su sei, e la
deviazione fra utenti è 0.013: **non sta scegliendo niente**. Spiega l'1.3%
della varianza di τ, mentre il 92.5% dipende da quale item si considera.

**Togliendo la divisione per √d** i pesi passano a 74.7/25.3, con deviazione
0.128 e valori fra 0.03 e 1.00. L'accuratezza non cambia (.0252 contro .0253).

La lettura: **la Figura 3 è coerente con una versione priva della divisione per
√d**, presumibilmente prodotta prima che quella normalizzazione entrasse nel
codice. Verificato su Elec/Cloth, non su Douban.

### 3.8 Il controfattuale: τ predice, ma il canale non serve

Azzerando il canale shared e ricalcolando le classifiche:

| | Elec→Cloth | Douban |
|---|---|---|
| Recall@20 intatto → ablato | .0202 → .0210 (**+4.0%**) | .0998 → .1036 (**+3.7%**) |
| top-20 che sopravvive | 92.3% | 91.6% |
| τ medio delle **cadute** | 0.4833 | 0.4542 |
| τ medio delle **rimaste** | 0.4160 | 0.3748 |
| differenza, p | +0.0673, **< 0.0001** | +0.0794, **< 0.0001** |

Due cose insieme. **τ è causalmente valida**: le raccomandazioni che crollano
sono quelle che τ aveva indicato, su due dataset. **E il canale non serve**:
toglierlo non costa accuratezza, semmai la migliora. Vale su tutto lo sweep di
`cl_org_weight` (+4.2%, +4.0%, +3.6%).

Nessuna ablazione, di nessun canale, sposta l'accuratezza oltre il 4% in nessuna
direzione.

### 3.9 τ come diagnostica di collasso

| | modello collassato | modello sano |
|---|---|---|
| τ medio | 0.4967 | 0.5031 |
| **deviazione** | **0.006** | **0.080** |

Il valore medio è quasi identico: **chi guarda solo la media conclude il
contrario del vero**. Il segnale è la dispersione. È una spia leggibile a colpo
d'occhio, definita sulla singola raccomandazione invece che sull'intero spazio
latente.

### 3.10 Le spiegazioni in linguaggio naturale funzionano, e sono fedeli

Lo stadio di verbalizzazione produce testo vincolato a non contraddire i numeri.
Un **secondo modello** (gemma4, famiglia diversa da qwen3.5 che scrive) legge
solo la spiegazione — niente numeri, niente cronologia — e stima il transfer
ratio. Su 40 spiegazioni:

| | |
|---|---|
| correlazione di Spearman | **+0.857** |
| correlazione di Pearson | +0.678 |
| errore assoluto medio | 0.087 |

E regge su un intervallo stretto (τ vero fra 0.131 e 0.590, deviazione 0.075).
**Il testo trasporta davvero il numero** — l'affermazione centrale del framework
è misurata, non asserita.

Nota: c'era un bug che rendeva vuote tutte le spiegazioni. Qwen3.5 ragiona prima
di rispondere e il ragionamento va in un campo separato, quindi con un budget di
token piccolo lo consumava tutto pensando. Ora il ragionamento è disattivato e
una risposta vuota solleva un errore invece di finire su disco in silenzio.

---

## 4. Cose ritirate

Vanno ricordate perché non riemergano:

- **"Il transfer va ai clienti abituali"** — la correlazione fra τ e la storia
  nel dominio target cambia segno al variare di un solo peso di loss: +0.067,
  +0.022, −0.013, −0.017. Artefatto degli iperparametri.
- **"Azzerare specific o base costa il 4%, azzerare shared no"** — vale su
  Elec→Cloth, non su Douban, dove anche specific migliora. L'asimmetria fra
  canali non è sostenibile.
- **"Ogni misura di indipendenza peggiora aumentando `cl_org_weight`"** — nasceva
  da un confronto confondato fra direzioni. Nello sweep pulito il probe è fermo e
  solo il gap di dCor si muove.
- **Tutto il Contributo 2** (audit semantico con LLM): lo strumento funzionava,
  ma la corrispondenza cross-domain si è dissolta con l'arrivo dei dati. Codice
  rimosso, resta una nota in `RISULTATI.md` §6.

---

## 5. Checkpoint disponibili

| file | dominio | note |
|---|---|---|
| `DGCDR-Jul-25-2026_11-32-27` | Elec→Cloth | **il riferimento**, org=0.1 |
| `DGCDR-Jul-28-2026_16-48-15` | Movie→Book | Douban |
| `DGCDR-Jul-28-2026_18-14-29` | Elec→Cloth | org=0.01 |
| `DGCDR-Jul-28-2026_19-38-50` | Elec→Cloth | org=10 |
| `DGCDR-Jul-31-2026_09-52-48` | Elec→Cloth | loss normalizzata |
| `DGCDR-Jul-31-2026_11-22-11` | Elec→Cloth | senza √d |
| `DGCDR-Jul-31-2026_12-45-43` | Elec→Cloth | entrambe |
| `DGCDR-Jul-25-2026_11-22-15` | CDs→Instruments | **collassato**, utile come controllo negativo |

Non caricabili: `DGCDR-Jul-25-2026_18-02-26` (allenato sulla VM, riferisce
directory con nomi diversi) e `DGCDR-Jul-26-2026_18-09-39` (aveva un
discriminatore poi rimosso).

Metadati: `item_metadata/cache_elec_cloth.json`, 135.217 item, risoluzione 100%.
I dump originali sono 23 GB e non servono più.

---

## 6. Limiti da dichiarare

- **Un seed per configurazione.** Niente è stato replicato con seed diversi. È
  il primo appunto che farà un revisore.
- **Sport&Cloth non misurata**: 149k utenti e 149k item, il training richiede
  ~11,7 GB di VRAM e non entra nei 12 disponibili. Restano fuori due delle sei
  barre della Figura 3.
- **Il √d verificato su una coppia sola.** Su Douban la Figura 3 dà Movie
  sbilanciato dall'altra parte, e non abbiamo verificato se anche quello si
  riproduca senza la divisione.
- **Il controfattuale** poggia su 300 utenti campionati per checkpoint.
- τ alto **non** significa raccomandazione migliore: è un'attribuzione, non una
  valutazione.
- Nelle spiegazioni generate compaiono piccole imprecisioni sui prodotti citati
  (un *"golf shirts"* al posto di *"golf shoes"*). Il vincolo numerico regge,
  quello sui prodotti meno.

---

## 7. Cosa si potrebbe fare

1. **Repliche con seed diversi** — ~1,5h per run su Elec→Cloth. Non insegnano
   nulla di nuovo ma chiudono l'obiezione più prevedibile.
2. **Douban senza √d**, per completare §3.7.
3. **Sport&Cloth** su hardware con più VRAM.
4. Un controllo sulle spiegazioni generate: verificare che ogni prodotto citato
   compaia davvero nella cronologia dell'utente.

---

## 8. In una frase

Il framework decompone le raccomandazioni cross-domain in modo esatto e
verificabile, τ è validata causalmente e le spiegazioni che produce trasportano
davvero i numeri. Applicandolo a DGCDR — su modelli che riproducono i risultati
pubblicati entro il 3% — emerge che **i meccanismi su cui il modello si regge
non fanno quello che l'articolo dichiara**, e che le correzioni ovvie non
bastano perché il problema non è nella formulazione dei vincoli ma in ciò che un
vincolo geometrico può ottenere.
