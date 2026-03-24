"""
DESCARTES-PHARMA Tier 1: Hodgkin-Huxley Simulator Ground Truth Generator

Generates training data with full biological intermediate variables.
Train an LSTM/GNN to replicate I->V mapping, then probe whether
hidden states recover m, h, n -- the zombie test with known answer.
"""

import numpy as np
from scipy.integrate import odeint


class HodgkinHuxleySimulator:
    """
    Generate ground truth data for DESCARTES-PHARMA validation.

    This is the UNIT TEST for the entire framework:
    1. Simulate HH neuron -> get (I, V, m, h, n) trajectories
    2. Train surrogate: I -> V (input-output only)
    3. Probe surrogate hidden states for m, h, n
    4. Known answer: good surrogate MUST encode m, h, n
    5. If probes fail to find m, h, n -> probe is broken
    6. If probes find m, h, n -> probe is validated
    """

    C_m = 1.0       # membrane capacitance (uF/cm^2)
    g_Na = 120.0    # sodium conductance (mS/cm^2)
    g_K = 36.0      # potassium conductance (mS/cm^2)
    g_L = 0.3       # leak conductance (mS/cm^2)
    E_Na = 50.0     # sodium reversal potential (mV)
    E_K = -77.0     # potassium reversal potential (mV)
    E_L = -54.387   # leak reversal potential (mV)

    @staticmethod
    def alpha_m(V):
        return 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0))

    @staticmethod
    def beta_m(V):
        return 4.0 * np.exp(-(V + 65.0) / 18.0)

    @staticmethod
    def alpha_h(V):
        return 0.07 * np.exp(-(V + 65.0) / 20.0)

    @staticmethod
    def beta_h(V):
        return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))

    @staticmethod
    def alpha_n(V):
        return 0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0))

    @staticmethod
    def beta_n(V):
        return 0.125 * np.exp(-(V + 65.0) / 80.0)

    def derivatives(self, state, t, I_func):
        V, m, h, n = state
        I = I_func(t)

        dVdt = (I - self.g_Na * m**3 * h * (V - self.E_Na)
                - self.g_K * n**4 * (V - self.E_K)
                - self.g_L * (V - self.E_L)) / self.C_m
        dmdt = self.alpha_m(V) * (1.0 - m) - self.beta_m(V) * m
        dhdt = self.alpha_h(V) * (1.0 - h) - self.beta_h(V) * h
        dndt = self.alpha_n(V) * (1.0 - n) - self.beta_n(V) * n

        return [dVdt, dmdt, dhdt, dndt]

    def simulate(self, I_func, T=100.0, dt=0.01):
        """
        Simulate HH neuron and return ALL biological variables.

        Returns dict with keys: t, V, m, h, n, I, I_Na, I_K, g_Na_eff, g_K_eff
        """
        t = np.arange(0, T, dt)
        V0, m0, h0, n0 = -65.0, 0.05, 0.6, 0.32

        solution = odeint(self.derivatives, [V0, m0, h0, n0], t,
                          args=(I_func,), hmax=dt)

        V = solution[:, 0]
        m = solution[:, 1]
        h = solution[:, 2]
        n = solution[:, 3]
        I = np.array([I_func(ti) for ti in t])

        g_Na_eff = self.g_Na * m**3 * h
        g_K_eff = self.g_K * n**4
        I_Na = g_Na_eff * (V - self.E_Na)
        I_K = g_K_eff * (V - self.E_K)

        return {
            't': t, 'V': V, 'm': m, 'h': h, 'n': n, 'I': I,
            'I_Na': I_Na, 'I_K': I_K,
            'g_Na_eff': g_Na_eff, 'g_K_eff': g_K_eff
        }

    def generate_dataset(self, n_trials=100, T=100.0, dt=0.01, seed=42):
        """
        Generate a full training dataset with varied input currents.

        Returns:
            dict with inputs, outputs, bio_targets, target_names, dt, T, n_trials
        """
        rng = np.random.default_rng(seed)
        t_steps = int(T / dt)

        inputs = np.zeros((n_trials, t_steps, 1))
        outputs = np.zeros((n_trials, t_steps, 1))
        bio_targets = np.zeros((n_trials, t_steps, 7))

        target_names = ['m', 'h', 'n', 'I_Na', 'I_K', 'g_Na_eff', 'g_K_eff']

        for i in range(n_trials):
            pattern_type = rng.choice(['step', 'ramp', 'noisy', 'pulse_train'])

            if pattern_type == 'step':
                amplitude = rng.uniform(0, 20)
                onset = rng.uniform(10, 30)
                I_func = lambda t, a=amplitude, o=onset: a if t > o else 0.0

            elif pattern_type == 'ramp':
                rate = rng.uniform(0.1, 0.5)
                I_func = lambda t, r=rate: r * t

            elif pattern_type == 'noisy':
                mean_I = rng.uniform(5, 15)
                noise_std = rng.uniform(1, 5)
                noise = rng.normal(mean_I, noise_std, t_steps)
                I_func = lambda t, n=noise, d=dt: n[min(int(t / d), len(n) - 1)]

            elif pattern_type == 'pulse_train':
                freq = rng.uniform(10, 100)
                amplitude = rng.uniform(10, 30)
                duty = rng.uniform(0.1, 0.5)
                I_func = lambda t, f=freq, a=amplitude, d=duty: \
                    a if (t * f / 1000) % 1.0 < d else 0.0

            result = self.simulate(I_func, T, dt)

            inputs[i, :, 0] = result['I']
            outputs[i, :, 0] = result['V']
            bio_targets[i, :, 0] = result['m']
            bio_targets[i, :, 1] = result['h']
            bio_targets[i, :, 2] = result['n']
            bio_targets[i, :, 3] = result['I_Na']
            bio_targets[i, :, 4] = result['I_K']
            bio_targets[i, :, 5] = result['g_Na_eff']
            bio_targets[i, :, 6] = result['g_K_eff']

        return {
            'inputs': inputs,
            'outputs': outputs,
            'bio_targets': bio_targets,
            'target_names': target_names,
            'dt': dt,
            'T': T,
            'n_trials': n_trials
        }
