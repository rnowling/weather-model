from collections import defaultdict
import random
import sys

import numpy as np
import matplotlib.pyplot as plt

from weather_model import fft, read_daily, simulate_temp, snowfall_rainfall_model, simulate_precip, simulate_wind_speed
from weather_model import wind_chill_model, wind_chill_transaction_probability, rainfall_transaction_probability
from weather_model import snowfall_transaction_probability, wind_speed_transaction_probability
from weather_model import weather_transaction_probability, autocorr

RECORDS = ["201110", "201111", "201112", "201201", "201202", "201203", "201204", "201205", "201206", "201207", "201208", 
	"201209", "201210", "201211", "201212", "201301", "201302", "201303", "201304", "201305", "201306", "201307", "201308",
	"201309", "201310", "201311", "201312", "201401", "201402", "201403", "201404", "201405", "201406", "201407", "201408",
	"201409"]

def plot_sim_autocorr(flname, temps, sim_temps):
	t_corr = autocorr(temps)
	st_corr = autocorr(sim_temps)
	plt.clf()
	plt.hold(True)
	plt.plot(t_corr, color="c", label="Real")
	plt.plot(st_corr, color="k", label="Simulated")
	plt.legend()
	plt.ylim([-1.0, 1.0])
	plt.xlabel("Time Lag (days)", fontsize=16)
	plt.ylabel("Correlation Coefficient", fontsize=16)
	plt.xlim([0, len(temps)])
	plt.savefig(flname, DPI=300)

def plot_sim_fft(flname, real, simulated):
	real_freq, real_amplitudes, _ = fft(autocorr(real))
	sim_freq, sim_amp, _ = fft(autocorr(simulated))

	n_real_samples = len(real)
	n_sim_samples = len(simulated)
	plt.clf()
	plt.hold(True)
	plt.plot(real_freq[1:(n_real_samples+1)/2], real_amplitudes[1:(n_real_samples+1)/2], color="c", label="Real")
	plt.plot(sim_freq[1:(n_sim_samples+1)/2], sim_amp[1:(n_sim_samples+1)/2], color="k", alpha=0.7, label="Simulated")
	plt.xlabel("Frequency (cycles/day)", fontsize=16)
	plt.ylabel("Amplitude", fontsize=16)
	plt.legend(loc="upper right")
	plt.xlim([0, 0.05])
	plt.savefig(flname, DPI=300)

def plot_sim_vel(flname, vel, sim_vel):
	plt.clf()
	plt.hold(True)
	plt.plot(sim_vel, color="k", label="Simulated")
	plt.plot(vel, color="c", label="Real")
	plt.xlabel("Time (Days)", fontsize=16)
	plt.ylabel("dT/dt (F)", fontsize=16)
	plt.legend()
	plt.xlim([0, len(vel)])
	plt.savefig(flname, DPI=300)

def plot_vel(flname, vel):
	plt.clf()
	plt.plot(vel, color="c")
	plt.xlabel("Time (Days)", fontsize=16)
	plt.ylabel("dT/dt (F)", fontsize=16)
	plt.xlim([0, len(vel)])
	plt.savefig(flname, DPI=300)

def plot_hist_sim_vel(flname, vel, sim_vel):
	plt.clf()
	plt.hold(True)
	plt.hist(vel, color="c", label="Real")
	plt.hist(sim_vel, color="k", label="Simulated", alpha=0.7)
	plt.ylabel("Occurrences (Days)", fontsize=16)
	plt.xlabel("dT/dt (F)", fontsize=16)
	plt.legend()
	plt.savefig(flname, DPI=300)

def plot_sim_temps(flname, temps, sim_temps):
	plt.clf()
	plt.hold(True)
	plt.plot(temps, color="c", label="Real")
	plt.plot(sim_temps, color="k", label="Simulated")
	plt.xlabel("Time (Days)", fontsize=16)
	plt.ylabel("Temperature (F)", fontsize=16)
	plt.xlim([0, len(temps)])
	plt.legend(fontsize=14)
	plt.savefig(flname, DPI=300)

def hist_temps(flname, temps, sim_temps):
	plt.clf()
	plt.hold(True)
	plt.hist(temps, bins=50, color="c", label="Real")
	plt.hist(sim_temps, bins=50, color="k", alpha=0.7, label="Simulated")
	plt.ylabel("Occurrences (Days)", fontsize=16)
	plt.xlabel("Temperature (F)", fontsize=16)
	plt.legend(loc="upper right")
	plt.savefig(flname, DPI=300)

def plot_precip(flname, precip, sim_precip):
	plt.clf()
	plt.hold(True)
	plt.plot(precip[:, 2], color="c", label="Real")
	plt.plot(sim_precip, color="k", label="Simulated")
	plt.xlabel("Time (Days)", fontsize=16)
	plt.ylabel("Total Precipitation (in)", fontsize=16)
	plt.legend()
	plt.savefig(flname, DPI=300)

def plot_hist_precip(flname, precip, sim_precip):
	plt.clf()
	plt.hold(True)

	bins = np.linspace(0.0, 2.0, 20.0)
	plt.hist(precip[:, 2], bins=bins, color="c", label="Real")
	plt.hist(sim_precip, bins=bins, color="k", alpha=0.7, label="Simulated")

	plt.ylabel("Occurrences (Days)", fontsize=16)
	plt.xlabel("Total Precipitation (in)", fontsize=16)
	plt.xlim([0.0, 2.0])
	plt.legend()
	plt.savefig(flname, DPI=300)

def plot_snowfall(flname, precip, sim_snowfall):
	plt.clf()
	plt.hold(True)
	plt.plot(precip[:, 0], color="c", alpha=0.7, label="Real")
	plt.plot(sim_snowfall, color="k", alpha=0.7, label="Simulated")
	plt.xlabel("Time (Days)", fontsize=16)
	plt.ylabel("Snowfall (in)", fontsize=16)
	plt.legend()
	plt.savefig(flname, DPI=300)

def plot_hist_snowfall(flname, precip, sim_snowfall):
	plt.clf()
	plt.hold(True)

	bins = np.linspace(0.0, 16.0, 32.0)
	plt.hist(precip[:, 0], bins=bins, color="c", alpha=0.7, label="Real")
	plt.hist(sim_snowfall, bins=bins, color="k", alpha=0.7, label="Simulated")

	plt.ylabel("Occurrences (Days)", fontsize=16)
	plt.xlabel("Snowfall (in)", fontsize=16)
	#plt.xlim([0.0, 2.0])
	plt.legend()
	plt.savefig(flname, DPI=300)

def plot_rainfall(flname, precip, sim_rainfall):
	plt.clf()
	plt.hold(True)
	plt.plot(precip[:, 1], color="c", alpha=0.7, label="Real")
	plt.plot(sim_rainfall, color="k", alpha=0.7, label="Simulated")
	plt.xlabel("Time (Days)", fontsize=16)
	plt.ylabel("Rainfall (in)", fontsize=16)
	plt.legend()
	plt.savefig(flname, DPI=300)

def plot_hist_rainfall(flname, precip, sim_rainfall):
	plt.clf()
	plt.hold(True)

	bins = np.linspace(0.0, 3.0, 30.0)
	plt.hist(precip[:, 1], bins=bins, color="c", alpha=0.7, label="Real")
	plt.hist(sim_rainfall, bins=bins, color="k", alpha=0.7, label="Simulated")

	plt.ylabel("Occurrences (Days)", fontsize=16)
	plt.xlabel("Rainfall (in)", fontsize=16)
	#plt.xlim([0.0, 2.0])
	plt.legend()
	plt.savefig(flname, DPI=300)

def plot_wind_chill_probability(flname):
	plt.clf()
	temps = np.linspace(-20.0, 90.0, 180.0)
	probability = wind_chill_transaction_probability(temps)
	plt.plot(temps, probability, color="c")
	plt.grid(True)
	plt.xlabel("Wind Chill (F)", fontsize=16)
	plt.ylabel("Probability", fontsize=16)
	plt.ylim([0.0, 1.0])
	plt.legend()
	plt.savefig(flname, DPI=300)

def plot_wind_chill_prob_hist(flname, temps, sim_temps):
	plt.clf()
	plt.hold(True)

	wc_prob = wind_chill_transaction_probability(temps)
	sim_wc_prob = wind_chill_transaction_probability(sim_temps)

	bins = np.linspace(0.0, 1.0, 20.0)
	plt.hist(wc_prob, bins=bins, color="c", label="Real")
	plt.hist(sim_wc_prob, bins=bins, color="k", alpha = 0.7, label="Simulated")

	plt.ylabel("Frequency", fontsize=16)
	plt.xlabel("Probability", fontsize=16)
	plt.xlim([0.0, 1.0])
	plt.grid(True)
	plt.legend(loc="upper left")
	plt.savefig(flname, DPI=300)

def plot_snowfall_trans_prob(flname):
	plt.clf()
	snowfall = np.linspace(0.0, 12.0, 60.0)
	quality = snowfall_transaction_probability(snowfall)
	plt.plot(snowfall, quality, color="c")
	plt.grid(True)
	plt.xlabel("Snowfall (in)", fontsize=16)
	plt.ylabel("Probability", fontsize=16)
	plt.ylim([0.0, 1.0])
	plt.savefig(flname, DPI=300)

def plot_hist_snowfall_trans_prob(flname, precip, sim_snowfall):
	plt.clf()
	plt.hold(True)

	bins = np.linspace(0.0, 1.0, 20.0)
	plt.hist(snowfall_transaction_probability(precip[:, 0]), bins=bins, color="c", alpha=0.7, label="Real")
	plt.hist(snowfall_transaction_probability(sim_snowfall), bins=bins, color="k", alpha=0.7, label="Simulated")

	plt.ylabel("Days (Frequency)", fontsize=16)
	plt.xlabel("Probability", fontsize=16)
	plt.xlim([0.0, 1.0])
	plt.grid(True)
	plt.legend(loc="upper left")
	plt.savefig(flname, DPI=300)

def plot_rainfall_trans_prob(flname):
	plt.clf()
	rainfall = np.linspace(0.0, 6.0, 60.0)
	prob = rainfall_transaction_probability(rainfall)
	plt.plot(rainfall, prob, color="c")
	plt.grid(True)
	plt.xlabel("Rainfall (in)", fontsize=16)
	plt.ylabel("Probability", fontsize=16)
	plt.ylim([0.0, 1.0])
	plt.savefig(flname, DPI=300)

def plot_hist_rainfall_trans_prob(flname, precip, sim_rainfall):
	plt.clf()
	plt.hold(True)

	bins = np.linspace(0.0, 1.0, 20.0)
	plt.hist(rainfall_transaction_probability(precip[:, 1]), bins=bins, color="c", alpha=0.7, label="Real")
	plt.hist(rainfall_transaction_probability(sim_rainfall), bins=bins, color="k", alpha=0.7, label="Simulated")

	plt.ylabel("Days (Frequency)", fontsize=16)
	plt.xlabel("Probability", fontsize=16)
	plt.xlim([0.0, 1.0])
	plt.grid(True)
	plt.legend(loc="upper left")
	plt.savefig(flname, DPI=300)

def plot_scatter_temp_precip(flname, temp, precip, a=0.4, b=27.0):
	plt.clf()
	percent = []
	for snowfall, rainfall, total_precip in precip:
		if total_precip > 0.0:
			percent.append(rainfall / total_precip)
		else:
			percent.append(0.0)
	temp_percent = np.array([(temp, precip) for temp, precip in zip(temp, percent) if precip > 0.0 and precip < 1.0])
	ts = np.linspace(0.0, 60.0, 120.0)
	predicted = [1.0 / (1.0 + np.exp(-a * (T - b))) for T in ts]
	plt.scatter(temp_percent[:, 0], temp_percent[:, 1])
	plt.hold(True)
	plt.plot(ts, predicted)
	plt.ylim([0.0, 1.0])
	plt.xlabel("Temperature (F)", fontsize=16)
	plt.ylabel("Rainfall (% of Precip)", fontsize=16)
	plt.savefig(flname, DPI=300)

def plot_wind_speed(flname, wind_speed, sim_wind_speed=None):
	plt.clf()
	plt.hold(True)
	plt.plot(wind_speed, color="c", label="Real")
	if sim_wind_speed != None:
		plt.plot(sim_wind_speed, color="k", label="Simulated")
	plt.xlabel("Time (Days)", fontsize=16)
	plt.ylabel("Wind Speed (mph)", fontsize=16)
	plt.legend()
	plt.savefig(flname, DPI=300)

def plot_hist_wind_speed(flname, wind_speed, sim_wind_speed=None):
	plt.clf()
	plt.hold(True)

	plt.hist(wind_speed, bins=30, color="c", alpha=0.7, label="Real")

	if sim_wind_speed != None:
		plt.hist(sim_wind_speed, bins=30, color="k", alpha=0.7, label="Simulated")

	plt.ylabel("Occurrences (Days)", fontsize=16)
	plt.xlabel("Wind Speed (mph)", fontsize=16)
	plt.legend()
	plt.savefig(flname, DPI=300)

def plot_wind_speed_fft(flname, real, simulated=None):
	real_freq, real_amplitudes, _ = fft(real)
	n_real_samples = len(real)
	plt.clf()
	plt.hold(True)
	plt.plot(real_freq[1:(n_real_samples+1)/2], real_amplitudes[1:(n_real_samples+1)/2], color="c", label="Real")
	
	if simulated != None:
		sim_freq, sim_amp, _ = fft(simulated)
		n_sim_samples = len(simulated)
		plt.plot(sim_freq[1:(n_sim_samples+1)/2], sim_amp[1:(n_sim_samples+1)/2], color="k", alpha=0.7, label="Simulated")

	plt.xlabel("Frequency (cycles/day)", fontsize=16)
	plt.ylabel("Amplitude", fontsize=16)
	plt.legend(loc="upper right")
	plt.xlim([0, 0.05])
	plt.savefig(flname, DPI=300)

def plot_wind_speed_trans_prob(flname):
	plt.clf()
	wind_speed = np.linspace(0.0, 30.0, 60.0)
	prob = wind_speed_transaction_probability(wind_speed)
	plt.plot(wind_speed, prob, color="c")
	plt.grid(True)
	plt.xlabel("Wind Speed (mph)", fontsize=16)
	plt.ylabel("Probability", fontsize=16)
	plt.ylim([0.0, 1.0])
	plt.savefig(flname, DPI=300)

def plot_hist_wind_speed_trans_prob(flname, wind_speed, sim_wind_speed):
	plt.clf()
	plt.hold(True)

	bins = np.linspace(0.0, 1.0, 20.0)
	plt.hist(wind_speed_transaction_probability(wind_speed), bins=bins, color="c", alpha=0.7, label="Real")
	plt.hist(wind_speed_transaction_probability(sim_wind_speed), bins=bins, color="k", alpha=0.7, label="Simulated")

	plt.ylabel("Days (Frequency)", fontsize=16)
	plt.xlabel("Probability", fontsize=16)
	plt.xlim([0.0, 1.0])
	plt.grid(True)
	plt.legend(loc="upper left")
	plt.savefig(flname, DPI=300)

def plot_weather_probability(flname, wind_chill, wind_speed, snowfall, rainfall, sim_wind_chill, sim_wind_speed, sim_snowfall, sim_rainfall):
	plt.clf()
	plt.hold(True)
	plt.plot(weather_transaction_probability(wind_chill, wind_speed, snowfall, rainfall), color="c", label="Real")
	plt.plot(weather_transaction_probability(sim_wind_chill, sim_wind_speed, sim_snowfall, sim_rainfall), color="k", alpha=0.7, label="Simulated")
	plt.xlabel("Time (Days)", fontsize=16)
	plt.ylabel("Probability", fontsize=16)
	plt.ylim([0.0, 1.0])
	plt.grid(True)
	plt.legend(loc="lower left")
	plt.savefig(flname, DPI=300)

def plot_weather_prob_hist(flname, wind_chill, wind_speed, snowfall, rainfall, sim_wind_chill, sim_wind_speed, sim_snowfall, sim_rainfall):
	plt.clf()
	plt.hold(True)
	bins = np.linspace(0.0, 1.0, 20.0)
	plt.hist(weather_transaction_probability(wind_chill, wind_speed, snowfall, rainfall), bins=bins, color="c", label="Real")
	plt.hist(weather_transaction_probability(sim_wind_chill, sim_wind_speed, sim_snowfall, sim_rainfall), bins=bins, color="k", alpha=0.7, label="Simulated")
	plt.xlabel("Occurrences (Days)", fontsize=16)
	plt.ylabel("Probability", fontsize=16)
	plt.xlim([0.0, 1.0])
	plt.grid(True)
	plt.legend(loc="upper left")
	plt.savefig(flname, DPI=300)

data_dir = sys.argv[1]
output_dir = sys.argv[2]

records = defaultdict(list)
for date in RECORDS:
	read_daily(data_dir + "/" + date + "daily.txt", records)

sbn_data = records["14848"] #records["12815"] 
sbn_data.sort()
temps = np.array([temp for (t, temp, _, _, _, _) in sbn_data])
vel = temps[1:] - temps[:len(temps) - 1]

#avg = np.average(temps)
std = 1.25 * np.std(vel)
freq, ampl, coeff = fft(temps)
fourier_coeff = coeff[3]
avg = 0.5 * np.abs(coeff)[0]
sim_temps =  simulate_temp(avg, std, 1.0, len(temps), fourier_coeff)
sim_vel = sim_temps[1:] - sim_temps[:len(sim_temps) - 1]

precip = np.array([(snowfall, rainfall, total_precip) for t, _, snowfall, rainfall, total_precip, _ in sbn_data])
sim_precip, sim_snowfall, sim_rainfall = simulate_precip(1.5 * np.average(precip[:, 2]), temps, precip.shape[0])

wind_speeds = np.array([wind_speed for _, _, _, _, _, wind_speed in sbn_data])
k = np.sqrt(np.var(wind_speeds))
theta = np.average(wind_speeds) / k
freq, ampl, coeff = fft(wind_speeds)
fourier_coeff = coeff[3]
sim_wind_speeds = simulate_wind_speed(0.0, k, theta, len(wind_speeds), fourier_coeff)

wind_chill = wind_chill_model(temps, wind_speeds)
sim_wind_chill = wind_chill_model(sim_temps, sim_wind_speeds)

hist_temps(output_dir + "/sim_temp_hist.pdf", temps, sim_temps)
plot_sim_temps(output_dir + "/sim_temp.pdf", temps, sim_temps)
plot_sim_vel(output_dir + "/sim_vel.pdf", vel, sim_vel)
plot_hist_sim_vel(output_dir + "/sim_vel_hist.pdf", vel, sim_vel)
plot_sim_autocorr(output_dir + "/sim_temp_autocorr.pdf", temps, sim_temps)
plot_sim_fft(output_dir + "/sim_temp_fft.pdf", temps, sim_temps)

plot_precip(output_dir + "/daily_precip.pdf", precip, sim_precip)
plot_snowfall(output_dir + "/daily_snowfall.pdf", precip, sim_snowfall)
plot_hist_precip(output_dir + "/daily_precip_hist.pdf", precip, sim_precip)
plot_hist_snowfall(output_dir + "/daily_snowfall_hist.pdf", precip, sim_snowfall)
plot_scatter_temp_precip(output_dir + "/daily_temp_precip_scatter.pdf", temps, precip, a=0.2)

plot_rainfall(output_dir + "/daily_rainfall.pdf", precip, sim_rainfall)
plot_hist_rainfall(output_dir + "/daily_rainfall_hist.pdf", precip, sim_rainfall)

plot_wind_speed(output_dir + "/daily_wind_speeds.pdf", wind_speeds, sim_wind_speeds)
plot_hist_wind_speed(output_dir + "/daily_wind_speed_hist.pdf", wind_speeds, sim_wind_speeds)
plot_wind_speed_fft(output_dir + "/daily_wind_speed_fft.pdf", wind_speeds, sim_wind_speeds)

plot_snowfall_trans_prob(output_dir + "/snowfall_trans_prob.pdf")
plot_hist_snowfall_trans_prob(output_dir + "/snowfall_trans_prob_hist.pdf", precip, sim_snowfall)

plot_rainfall_trans_prob(output_dir + "/rainfall_trans_prob.pdf")
plot_hist_rainfall_trans_prob(output_dir + "/rainfall_trans_prob_hist.pdf", precip, sim_rainfall)

plot_scatter_temp_precip(output_dir + "/daily_wind_chill_precip_scatter.pdf", wind_chill, precip, a=0.2, b=20.0)

plot_wind_chill_probability(output_dir + "/wind_chill_trans_prob.pdf")
plot_wind_chill_prob_hist(output_dir + "/wind_chill_trans_prob_hist.pdf", wind_chill, sim_wind_chill)

plot_wind_speed_trans_prob(output_dir + "/wind_speed_trans_prob.pdf")
plot_hist_wind_speed_trans_prob(output_dir + "/wind_speed_trans_prob_hist.pdf", wind_speeds, sim_wind_speeds)

plot_weather_probability(output_dir + "/weather_trans_prob.pdf", wind_chill, wind_speeds, precip[:, 0], precip[:, 1], sim_wind_chill, sim_wind_speeds, sim_snowfall, sim_rainfall)
plot_weather_prob_hist(output_dir + "/weather_trans_prob_hist.pdf", wind_chill, wind_speeds, precip[:, 0], precip[:, 1], sim_wind_chill, sim_wind_speeds, sim_snowfall, sim_rainfall)

