# Projekt_SI_Studia


load_data_test: 
  Funkcja do zwracająca danych testowych do programu zczytując po kolei nazwy plików z pliku .csv w postaci zdjęć .png. Dane są labelowane na przejścia dla pieszych: 1, oraz nie przejścia dla pieszych: 0, za pomocą zczytywania danych z pliku .xml
  
  
load_data_train:
  Funkcja spełniająca tą samą rolę co funkcja load_data_test, tylko ta używana jest do trenowania, a więc zwraca tylko i wyłączenie tablicę zdjęć ze znakami z przejściem dla         pieszych. Aby tego dokonać wykorzystywany jest plik .xml, z którego zczytywana jest nazwa zdjęcia.


learn_bovw:
   Funkcja do nauki słownika BoVW, który zapisywany jest w pliku  "voc.npy".
  
  
extract_features:
  Funkcja odpowiedzialna za ekstrakcję cech i oznaczająca je jako 'desc'
  
  
train:
  Funkcja odpowiadająca za wytrenowanie wykorzystująca klasyfikację Random Forest
  
  
predict:
  Funkcja do wykrywanie czy dane wejściowe są znakami przejścia dla pieszych czy nie na podstawie wcześniej wyuczonego modelu. Dodatkowo funkcja posiada wyświetlanie nazwy pliku   testowego oraz (prawdopodobnie)  zakomentowane działające wyświetlanie ilości znaków przejść dla pieszych na zdjęciu oraz wyświetlanie współrzędnych prostokąta w którym się     znajduje pierwszy wykryty znak - do ich działania potrzebna jest biblioteka cvlib oraz zainstalowana tensorflow. W przypadku tej 2 występował u mnie błąd w postaci zbyt     długiej  ścieżki, mimo próby rozwiązania problemu drogą wskazaną innym posiadaczą tego problemu na forach nie udało mi się rozwiązać z tego powodu nie byłem w stanie  zweryfikować czy owe   zakomentowane rozwiązanie jest prawidłowe.
  
  
 balance_dataset:
  Zwraca znormalizowane dane
