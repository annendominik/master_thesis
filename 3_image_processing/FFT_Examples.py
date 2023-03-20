#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Import modules
import numpy as np
import matplotlib.pyplot as plt

# Construct a time signal
Fs = 2000 # Sampling frequence
tstep = 1 / Fs # sample time interval
f0 = 100 # signal frequence
f1 = 900 # noise frequence
f2 = 50
f3 = 500

N = int(10 * Fs / f0) # number of samples

t = np.linspace(0, (N-1)*tstep, N) # time steps
fstep = Fs / N # freq interval
f = np.linspace(0, (N-1)*fstep, N) # freq step

y1 = 1 * np.sin(2 * np.pi * f0 * t)
y2 = 1 * np.sin(2 * np.pi * f0 * t) + 0.25 * np.sin(2 * np.pi * f1 * t)
y3 = 1 * np.sin(2 * np.pi * f0 * t) + 0.25 * np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t) + 0.75 * np.sin(2 * np.pi * f3 * t)

# perform fft
X1 = np.fft.fft(y1)
X2 = np.fft.fft(y2)
X3 = np.fft.fft(y3)
X_mag1 = np.abs(X1) / N
X_mag2 = np.abs(X2) / N
X_mag3 = np.abs(X3) / N

f_plot = f[0:int(N/2+1)]
X_mag_plot1 = 2 * X_mag1[0:int(N/2+1)]
X_mag_plot2 = 2 * X_mag2[0:int(N/2+1)]
X_mag_plot3 = 2 * X_mag3[0:int(N/2+1)]

X_mag_plot1[0] = X_mag_plot1[0] / 2
X_mag_plot2[0] = X_mag_plot2[0] / 2
X_mag_plot3[0] = X_mag_plot3[0] / 2

#plot1
fig, [ax1, ax2] = plt.subplots(nrows=2, ncols=1)
ax1.plot(t, y1)
ax2.plot(f_plot, X_mag_plot1, 'green')
ax1.set_xlabel("Time [s]")
ax2.set_xlabel("Frequence [Hz]")
ax1.grid()
ax2.grid()

ax1.set_xlim(0, t[-1])
ax2.set_xlim(0, f_plot[-1])
plt.tight_layout()
plt.show()

#plot2
fig, [ax1, ax2] = plt.subplots(nrows=2, ncols=1)
ax1.plot(t, y2)
ax2.plot(f_plot, X_mag_plot2, 'green')
ax1.set_xlabel("Time [s]")
ax2.set_xlabel("Frequence [Hz]")
ax1.grid()
ax2.grid()

ax1.set_xlim(0, t[-1])
ax2.set_xlim(0, f_plot[-1])
plt.tight_layout()
plt.show()

#plot3
fig, [ax1, ax2] = plt.subplots(nrows=2, ncols=1)
ax1.plot(t, y3)
ax2.plot(f_plot, X_mag_plot3, 'green')
ax1.set_xlabel("Time [s]")
ax2.set_xlabel("Frequence [Hz]")
ax1.grid()
ax2.grid()

ax1.set_xlim(0, t[-1])
ax2.set_xlim(0, f_plot[-1])
plt.tight_layout()
plt.show()
