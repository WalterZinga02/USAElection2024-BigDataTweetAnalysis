# BigData

Script Python per la pulizia e il preprocessing di dataset di tweet relativi agli swing states statunitensi.

## Avvio

1. Crea e attiva un ambiente virtuale Python.
2. Installa le dipendenze:

   ```powershell
   pip install pandas ftfy streamlit plotly
   ```

3. Inserisci i file CSV originali nella cartella `RawData`.
4. Esegui:

   ```powershell
   python preprocessing.py
   ```

I file puliti vengono creati nella cartella `CleanData` con il suffisso `_cleaned`.

## Dashboard Streamlit

Dopo aver generato i risultati con `spark_analysis.py`, avvia l'interfaccia grafica:

```powershell
streamlit run streamlit_app.py
```

La dashboard legge i CSV nella cartella `output`, mostra grafici interattivi e usa mappe colorate per le distribuzioni geografiche per stato.

Nel pannello laterale puoi anche avviare direttamente le query Spark: scegli input
e cartella output, poi premi il bottone della singola query da eseguire. Il log
viene mostrato nell'interfaccia e, a fine esecuzione, i grafici vengono ricaricati
dai nuovi CSV.

## Etichettatura con Ollama

Dopo aver avviato Ollama e scaricato il modello, ad esempio `ollama pull llama3`, esegui:

```powershell
pip install requests
python label_tweets_ollama.py
```

Lo script legge e sovrascrive tutti i CSV in `CleanData`, aggiungendo la
sola colonna `binary_label`: `1` indica un tweet
pro Trump, `0` un tweet pro Harris; le celle vuote rappresentano tweet neutrali,
ambigui o non classificabili e vanno escluse dall'addestramento della SVM binaria.
