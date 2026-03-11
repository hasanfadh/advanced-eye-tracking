def verify_fix():
    """
    Jalankan fungsi ini untuk memastikan fix berhasil.
    Buat file verify_freq.py di root folder, isi dengan kode ini,
    lalu jalankan: python verify_freq.py
    """
    import numpy as np
    import sys, os
    sys.path.append(os.path.dirname(__file__))
    from signal_processing.frequency_domain import FrequencyAnalyzer

    analyzer = FrequencyAnalyzer(sampling_rate=30)

    # Simulasi sinyal 90 frames dengan komponen frekuensi rendah
    t = np.linspace(0, 3, 90)  # 3 detik, 90 samples
    # Sinyal dengan 3 komponen: 0.3 Hz (very_low), 1.0 Hz (low), 3.0 Hz (medium)
    test_signal = (
        0.5 * np.sin(2 * np.pi * 0.3 * t) +   # should appear in very_low
        0.3 * np.sin(2 * np.pi * 1.0 * t) +   # should appear in low
        0.1 * np.sin(2 * np.pi * 3.0 * t)     # should appear in medium
    )

    features = analyzer.extract_spectral_features(test_signal)
    bands = features.get('frequency_bands', {})

    print("\n=== VERIFIKASI FIX FREQUENCY BAND ===")
    print(f"Resolusi frekuensi: {30/90:.3f} Hz per bin")
    print()

    expected = {'very_low': True, 'low': True, 'medium': True}
    all_pass = True

    for band_name, b in bands.items():
        rel = b.get('relative_power', 0)
        has_power = rel > 0.5  # minimal 0.5% power
        status = "✓ PASS" if has_power == expected.get(band_name, False) else "✗ FAIL"
        if status == "✗ FAIL":
            all_pass = False
        print(f"  {band_name:<12} rel={rel:.2f}%  {status}")

    print()
    if all_pass:
        print("✓ Fix berhasil! Band 0–0.5Hz dan 0.5–2Hz sekarang terdeteksi.")
    else:
        print("✗ Fix belum berhasil. Cek kembali perubahan di compute_psd().")

    return all_pass


if __name__ == "__main__":
    verify_fix()