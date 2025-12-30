# 🧠 RLveZS 
(production_ai_project hazırlık deneme yanılma öğrenme) 
## Pekiştirmeli Öğrenme (Reinforcement Learning) ve Zaman Serisi (Time Series) Entegrasyonu

Bu proje, **zaman serisi tahminleri ile pekiştirmeli öğrenmeyi birleştirerek** karar verme problemlerine çözüm üretmeyi amaçlayan deneysel ve modüler bir çalışmadır.  
Amaç yalnızca tahmin yapmak değil, **tahminleri aksiyona dönüştüren bir RL ajanı** tasarlamaktır.

---

## 🎯 Projenin Amacı

Klasik zaman serisi modelleri *“ne olacak?”* sorusuna cevap verirken,  
pekiştirmeli öğrenme *“ne yapmalıyım?”* sorusuna odaklanır.

Bu projede:

- 📈 **Zaman serisi modelleri** ile geleceğe yönelik öngörüler üretilir  
- 🎮 **RL ajanı**, bu öngörüleri kullanarak en iyi aksiyonu öğrenir  
- 🔁 Tahmin + karar verme **uçtan uca** bir yapıya dönüştürülür

---

## 🧩 Kullanılan Yaklaşım

### 1) Zaman Serisi (ZS)
- Simüle edilmiş veya gerçek veriler kullanılır  
- Trend, sezonsallık ve gürültü bileşenleri analiz edilir  
- Tahmin çıktıları **durum (state)** veya **ek girdi** olarak RL ortamına aktarılır  

### 2) Pekiştirmeli Öğrenme (RL)
- Ortam (Environment) zaman serisi dinamiklerine göre tanımlanır  
- Ajan:
  - Gözlem (state)  
  - Aksiyon (action)  
  - Ödül (reward)  
  - Politika (policy)  
- Amaç: **uzun vadeli toplam ödülü maksimize etmek**

---

## 🗂️ Proje Dizini

```
RLveZS/
│
├── dashboard/              # Görselleştirme ve izleme panelleri
├── data/
│   └── simulated/          # Simüle edilmiş zaman serisi verileri
│
├── models/                 # Eğitilmiş modeller / ağırlıklar
├── notebooks/              # Deneyler, analizler, prototipler
├── reports/
│   └── figures/            # Grafikler ve sonuç görselleri
│
├── src/                    # Ana kaynak kodlar
├── sirket.py               # Örnek senaryo / firma bazlı simülasyon
├── requirements.txt        # Bağımlılıklar
└── .gitignore
```

---

## ⚙️ Kurulum

```bash
git clone https://github.com/busraminal/RLveZS.git
cd RLveZS
pip install -r requirements.txt
```

> Python 3.9+ önerilir.

---

## 🚀 Çalıştırma Akışı (Özet)

1. **Veri Hazırlığı**
   - `data/simulated/` altındaki zaman serisi verileri yüklenir  

2. **Zaman Serisi Analizi**
   - Notebook’larda tahminleme ve görselleştirme yapılır  

3. **RL Ortamı**
   - Zaman serisi çıktıları ortam durumuna dahil edilir  

4. **Ajan Eğitimi**
   - Ödül fonksiyonu üzerinden öğrenme sağlanır  

5. **Sonuç Analizi**
   - `reports/figures/` altında performans grafikleri incelenir  

---

## 📊 Örnek Senaryo

- Talep tahmini yapılan bir sistem  
- RL ajanı:
  - Üretimi artır / azalt  
  - Stok tut / tutma  
- Yanlış karar → ceza  
- Doğru zamanlama → ödül  

Bu yapı:
- Üretim planlama  
- Enerji yönetimi  
- Finansal portföy  
- Stok & tedarik zinciri  
gibi alanlara uyarlanabilir.

---

## 📌 Öne Çıkan Özellikler

- ✅ Zaman serisi + RL **entegrasyonu**
- ✅ Modüler ve genişletilebilir yapı
- ✅ Notebook tabanlı deneysel çalışma
- ✅ Akademik projelere uygun mimari

---

## 🔮 Geliştirme Planları

- [ ] Gerçek veri setleri ile test  
- [ ] LSTM / Transformer tabanlı ZS modelleri  
- [ ] PPO / DQN gibi farklı RL algoritmaları  
- [ ] Dashboard üzerinden canlı izleme  
- [ ] Detaylı deney raporları  

---

## 👩‍💻 Geliştirici

**Büşra Mina AL**  
Yapay Zeka Mühendisliği & Endüstri Mühendisliği  
📌 Reinforcement Learning • Time Series 

---

## ⚠️ Not
Bu repo **araştırma ve eğitim amaçlıdır**.  
Notebook’lar deneysel olup sürekli geliştirilmektedir.

