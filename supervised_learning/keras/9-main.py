#!/usr/bin/env python3
import tensorflow as tf

# --- SƏNİN YAZMALI OLDUĞUN FUNKSİYALAR ---

def save_model(network, filename):
    """Modeli tam olaraq fayla yazır"""
    network.save(filename)

def load_model(filename):
    """Modeli fayldan geri yükləyir"""
    return tf.keras.models.load_model(filename)


# --- TEST HİSSƏSİ (Yoxlamaq üçün) ---
if __name__ == '__main__':
    # Sadə bir test modeli yaradırıq
    inputs = tf.keras.Input(shape=(10,))
    outputs = tf.keras.layers.Dense(3)(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Test datası ilə yoxlayırıq
    data = tf.random.normal((1, 10))
    print("Orijinal modelin nəticəsi:")
    print(model(data).numpy())

    # Modeli yadda saxlayırıq
    save_model(model, 'network.keras')
    print("\nModel 'network.keras' olaraq yadda saxlanıldı.")

    # Modeli yenidən yükləyirik
    loaded_net = load_model('network.keras')
    print("\nModel fayldan geri yükləndi.")
    
    print("\nYüklənmiş modelin nəticəsi (eyni olmalıdır):")
    print(loaded_net(data).numpy())
