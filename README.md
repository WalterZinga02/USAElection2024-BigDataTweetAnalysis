# BigData

Script Python per la pulizia e il preprocessing di dataset di tweet relativi agli swing states statunitensi.

## Avvio

1. Crea e attiva un ambiente virtuale Python.
2. Installa le dipendenze:

   ```powershell
   pip install pandas ftfy
   ```

3. Inserisci i file CSV nella cartella `RawData`.
4. Esegui:

   ```powershell
   python preprocessing.py
   ```

I file puliti vengono creati nella cartella `RawData` con il suffisso `_cleaned`.
