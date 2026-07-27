# Contributo 2 — il piano

Aggiornato il 26 luglio 2026, dopo la Fase 1. Branch `explainability-2`.

La prima versione di questo piano è stata in gran parte falsificata dai suoi
stessi cancelli. Quella che segue è la versione che resta in piedi.

---

## 1. L'idea di partenza

Il modello divide le preferenze in canale **shared** e canale **specific**. Ma
sono due vettori: nessuno sa cosa codifichino davvero.

L'idea era usare un LLM come **strumento di misura semantica**. Si prendono gli
item che il modello colloca vicini in un sottospazio, si mostrano titoli e
categorie all'LLM, e gli si chiede quale concetto abbiano in comune.

Poi si confrontano i concetti dei due canali. Se il disentanglement funziona, dal
canale shared devono uscire concetti generici ("regali per principianti",
"fascia alta") e dal canale specific concetti legati al dominio ("corde per
chitarra").

---

## 2. Perché il piano è cambiato

Prima di coinvolgere l'LLM abbiamo verificato una precondizione: **i due canali
sono davvero diversi?**

No, e in modo netto. I due canali item inducono la stessa geometria — la
correlazione tra le matrici di similarità item-item è **+0.9984**, con null a
+0.0002. Mettono vicini gli stessi item e lontani gli stessi item.

La causa è che i due gate hanno imparato la stessa funzione: si sovrappongono al
**98.2%**, con medie 0.849 e 0.852.

**Quindi il confronto previsto non ha più oggetto.** L'LLM avrebbe prodotto due
liste di concetti identiche, e la misura avrebbe dato zero per costruzione —
senza che quello zero significasse niente.

Il cancello ha fatto il suo lavoro: ha fermato il piano **prima** di spendere ore
di chiamate LLM su un confronto inesistente.

---

## 3. Cosa faremo invece

Resta in piedi una domanda diversa, non toccata dal problema qui sopra.

DGCDR ha una seconda regola oltre all'ortogonalità: `cl_sim_weight`, la loss che
**allinea i canali shared dei due domini**. È l'unico meccanismo del modello che
punta all'invarianza di dominio.

> **La domanda: quell'allineamento è semanticamente reale, o è una coincidenza
> geometrica?**

Se dal dominio source esce il concetto "rock anni '70", dal target dovrebbe
uscire qualcosa come "chitarre elettriche e amplificatori valvolari". Se i
concetti dei due domini non si accoppiano, la loss non sta ottenendo nulla di
sostanziale.

La domanda confronta i **due domini**, non i due canali, quindi il collasso dei
gate non la tocca. Ed è anche l'unica rimasta a cui **solo un LLM** può
rispondere: bisogna giudicare se due insiemi di item completamente diversi
parlino, in fondo, della stessa cosa.

---

## 4. Come si fa, passo per passo

### Passo 1 — Estrarre i cluster

Per ciascun dominio si raggruppano gli item vicini nel sottospazio, con k-means
sulla sfera unitaria (geometria coseno, così gli item popolari non formano
cluster propri per via della norma maggiore). Già implementato in `concepts.py`.

### Passo 2 — Nominare i cluster, alla cieca

Il prompt riceve **solo** titoli e categorie, mescolati. Mai l'ID interno, il
nome del dominio, il nome del canale o la posizione nel ranking. L'LLM non deve
poter indovinare cosa sta guardando.

Deve poter rispondere "nessun tema comune": quella risposta è un dato, non un
fallimento.

### Passo 3 — Validare le etichette *(cancello)*

Un'etichetta plausibile non è un'etichetta corretta. Quindi:

1. si tengono da parte alcuni item del cluster, mai mostrati durante il naming;
2. si mostra all'LLM l'etichetta e *N* insiemi di item, uno vero e gli altri da
   cluster diversi;
3. gli si chiede quale corrisponde.

Accuratezza sopra 1/N significa che l'etichetta cattura davvero la direzione
latente. **Se fallisce ci si ferma qui**: lo strumento non misura, e tutto quello
che verrebbe dopo sarebbe aneddotico.

### Passo 4 — La corrispondenza cross-domain *(risultato principale)*

Si estraggono e si nominano i concetti dei due domini separatamente, poi un
**secondo** LLM — diverso da quello che ha nominato, per evitare che si dia
ragione da solo — li accoppia.

Il risultato va letto contro due riferimenti:

| riferimento | cosa dice |
|---|---|
| accoppiamento casuale | il null: quanto si ottiene per caso |
| stessa misura sul canale **base** | quanto si otterrebbe **senza** disentanglement |

Il secondo è quello che conta. Il canale base è l'embedding GNN grezzo, senza
alcuna separazione. Se i concetti shared non si accoppiano meglio di quelli base,
**la loss di allineamento non ha aggiunto nulla di semantico**: la corrispondenza
c'era già nella struttura collaborativa.

### Passo 5 — Mappa concettuale descrittiva

Sugli item esiste una sola struttura, ma dire **cosa** codifica resta utile. Non
è un test, è una descrizione — la risposta leggibile alla domanda "cosa
rappresenta davvero questo modello".

---

## 5. Prerequisiti

| | stato |
|---|---|
| Checkpoint sano con `item_disentangle=True` | ✅ Elec→Cloth, in locale |
| Metadati Elec + Cloth | ✅ in cache, 135.217 item, risoluzione 100% |
| Risoluzione ID → token | ✅ verificata |
| Estrazione dei cluster | ✅ `concepts.py` |
| Endpoint LLM | ⚠️ **manca solo questo** |

Il vincolo dei metadati, che nella prima versione bloccava tutto, è risolto: il
caricamento filtrato in streaming impiega 38 secondi sui 23 GB di dump.

Nota: il bug di mappatura ID in `dgcdr.py` **non è un prerequisito**. Riguarda il
caricamento degli embedding testuali dentro il modello, cioè la semantic loss in
training. L'audit legge i metadati tramite `metadata.py`, che ha già la mappatura
corretta.

---

## 6. Cosa resta da implementare

**`concept_naming.py`** — etichettatura cieca, con i vincoli del Passo 2.

**`concept_eval.py`** — `label_validity` (Passo 3) e `cross_domain_matching`
(Passo 4), ciascuna con il proprio null.

**`audit_concepts.py`** — CLI sul modello di `explain_dgcdr.py`. Deve rifiutarsi
di girare su canali collassati, e il controllo va fatto **per canale**: sugli
item di questo checkpoint la correlazione tra strutture è 0.9984, quindi lo
script deve dirlo invece di procedere in silenzio.

---

## 7. Come può fallire

Vale la pena scriverlo prima, non dopo.

- **Le etichette non superano il Passo 3** → lo strumento non misura, ci si ferma.
- **I concetti shared si accoppiano come quelli base** → l'allineamento non
  aggiunge nulla. È il risultato che mi aspetto, ed è pubblicabile.
- **Nessuno dei due si accoppia sopra il caso** → i sottospazi non hanno alcuna
  corrispondenza cross-domain. Risultato più forte, ma va escluso che dipenda da
  cluster incoerenti: per questo il Passo 3 viene prima.
- **Circolarità** → chi nomina e chi accoppia devono essere due LLM diversi, ed
  entrambi ciechi sulla provenienza dei dati.
- **Contaminazione da popolarità** → i cluster potrebbero riflettere la
  popolarità invece del contenuto. Da controllare correlando la dimensione dei
  cluster con la popolarità media degli item.

---

## 8. Una nota onesta sul ruolo dell'LLM

I risultati più forti raccolti finora — collasso dei gate item, struttura
identica dei due canali, irremovibilità lineare dell'informazione di dominio —
**non hanno richiesto alcun LLM**. Sono misure di algebra lineare, ciascuna con
il proprio null.

Il Contributo 2 resta l'unico punto in cui un LLM fa qualcosa che nessuna misura
numerica può fare: leggere il contenuto degli item e giudicare se due insiemi di
concetti, estratti da domini diversi, parlino della stessa cosa. Vale la pena
farlo, ma con l'aspettativa giusta: è una misura in più, non la spina dorsale del
lavoro.
