import numpy as np
import matplotlib.pyplot as plt

x0 = 0.4
print(f"cos({x0}) = {np.cos(x0):.10f}")

k_values = np.arange(1, 11)

def experimento(dtype):
    residuos = []
    for k in k_values:
        a = np.array([1, 0, 0], dtype=dtype)
        b = np.array([1, 10.0**(-k), 0], dtype=dtype)

        c = np.cross(a, b)
        residuo = np.abs(np.dot(a, c))
        residuos.append(residuo)
    return np.array(residuos, dtype=dtype)

res_f32 = experimento(np.float32)
res_f64 = experimento(np.float64)

# 3 tabla
print(f"\n{'k':>3} | {'|a·c| float32':>15} | {'|a·c| float64':>15}")
print("-" * 40)
for k, r32, r64 in zip(k_values, res_f32, res_f64):
    print(f"{k:>3} | {r32:>15.30e} | {r64:>15.30e}")

# 4 grafico semilogaritmico
eps32 = np.finfo(np.float32).tiny
eps64 = np.finfo(np.float64).tiny
res_f32_plot = np.where(res_f32 == 0, eps32, res_f32)
res_f64_plot = np.where(res_f64 == 0, eps64, res_f64)

plt.figure(figsize=(8, 5.5))
plt.semilogy(k_values, res_f32_plot, 'o-', label='float32', color='tab:orange')
plt.semilogy(k_values, res_f64_plot, 's-', label='float64', color='tab:blue')
plt.xlabel('k  (b = (1, 10⁻ᵏ, 0))')
plt.ylabel(r'Residuo  $|\vec{a}\cdot(\vec{a}\times\vec{b})|$')
plt.title('Ortogonalidad numérica del producto vectorial vs. k')
plt.grid(True, which='both', alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('residuo_ortogonalidad.png', dpi=150)
plt.show()
