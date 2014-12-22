from collections import defaultdict
import random
import sys

import numpy as np
import matplotlib.pyplot as plt

from weather_model import fft, read_daily, simulate_temp, snowfall_rainfall_model, simulate_precip, simulate_wind_speed
from weather_model import weather_quality_snow, weather_quality_temp

RECORDS = ["201110", "201111", "201112", "201201", "201202", "201203", "201204", "201205", "201206", "201207", "201208", 
	"201209", "201210", "201211", "201212", "201301", "201302", "201303", "201304", "201305", "201306", "201307", "201308",
	"201309", "201310", "201311", "201312", "201401", "201402", "201403", "201404", "201405", "201406", "201407", "201408",
	"201410"]

def autocorr(series):
	shifted = series - np.average(series)
	corr = np.correlate(shifted, shifted, "full")
	corr2 = corr[:len(corr)/2] / np.var(shifted) / float(len(corr))
	return corr2

def plot_autocorr(flname, series):
	corr = autocorr(series)
	plt.clf()
	plt.plot(corr, color="c")
	plt.ylim([-1.0, 1.0])
	plt.xlabel("Time Lag (days)", fontsize=16)
	plt.ylabel("Correlation Coefficient", fontsize=16)
	plt.xlim([0, len(series)])
	plt.savefig(flname, DPI=300)

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

def plot_fft(flname, series):
	freq, amplitudes = fft(series)

	n_samples = len(series)
	plt.clf()
	plt.plot(freq[1:(n_samples+1)/2], amplitudes[1:(n_samples+1)/2], color="c")
	plt.xlabel("Frequency (cycles/day)", fontsize=16)
	plt.ylabel("Amplitude", fontsize=16)
	plt.xlim([0, 0.05])
	plt.savefig(flname, DPI=300)

def plot_sim_fft(flname, real, simulated):
	real_freq, real_amplitudes = fft(real)
	sim_freq, sim_amp = fft(simulated)

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

def plot_temps(flname, temps):
	plt.clf()
	plt.plot(temps, color="c")
	plt.xlabel("Time (Days)", fontsize=16)
	plt.ylabel("Temperature (F)", fontsize=16)
	plt.xlim([0, len(temps)])
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

def plot_hist_vel(flname, vel):
	plt.clf()
	plt.hist(vel, color="c")
	plt.ylabel("Occurrences (Days)", fontsize=16)
	plt.xlabel("dT/dt (F)", fontsize=16)
	#plt.xlim([0, len(vel)])
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

def hist_temps(flname, temps):
	plt.clf()
	plt.hist(temps, bins=50, color="c")
	plt.ylabel("Occurrences (Days)", fontsize=16)
	plt.xlabel("Temperature (F)", fontsize=16)
	plt.savefig(flname, DPI=300)

def plot_weather_quality_temp(flname):
	plt.clf()
	temps = np.linspace(0.0, 90.0, 180.0)
	quality = weather_quality_temp(temps)
	plt.plot(temps, quality, color="r")
	plt.grid(True)
	plt.xlabel("Temperature (F)", fontsize=16)
	plt.ylabel("Quality", fontsize=16)
	plt.ylim([0.0, 1.0])
	plt.legend()
	plt.savefig(flname, DPI=300)

def plot_hist_quality_temp(flname, temps, sim_temps):
	plt.clf()
	plt.hold(True)

	counts, bins = np.histogram(weather_quality_temp(temps), bins=10)
	counts = counts / float(sum(counts))
	plt.plot(bins[:-1], counts, color="r", label="Observed")

	counts, bins = np.histogram(weather_quality_temp(sim_temps), bins=10)
	counts = counts / float(sum(counts))
	plt.plot(bins[:-1], counts, color="g", label="Simulated")

	plt.ylabel("Days (Frequency)", fontsize=16)
	plt.xlabel("Quality", fontsize=16)
	plt.xlim([0.0, 1.0])
	plt.grid(True)
	plt.legend(loc="upper left")
	plt.savefig(flname, DPI=300)

def plot_precip(flname, precip, sim_precip):
	plt.clf()
	plt.hold(True)
	plt.plot(precip[:, 2], color="r", label="Observed")
	plt.plot(sim_precip, color="g", label="Simulated")
	plt.xlabel("Time (Days)", fontsize=16)
	plt.ylabel("Total Precipitation (in)", fontsize=16)
	plt.legend()
	plt.savefig(flname, DPI=300)

def plot_hist_precip(flname, precip, sim_precip):
	plt.clf()
	plt.hold(True)

	counts, bins = np.histogram(precip[:, 2], bins=30)
	plt.plot(bins[:-1], counts, color="r", label="Observed")

	counts, bins = np.histogram(sim_precip, bins=30)
	plt.plot(bins[:-1], counts, color="g", label="Simulated")

	plt.ylabel("Occurrences (Days)", fontsize=16)
	plt.xlabel("Total Precipitation (in)", fontsize=16)
	plt.xlim([0.0, 2.0])
	plt.legend()
	plt.savefig(flname, DPI=300)

def plot_scatter_precip(flname, precip):
	plt.clf()
	plt.hold(True)
	plt.scatter(precip[:, 1], precip[:, 0])
	plt.xlabel("Rainfall (in)", fontsize=16)
	plt.ylabel("Snowfall (in)", fontsize=16)
	plt.savefig(flname, DPI=300)

def plot_hist_snowfall(flname, precip, sim_snowfall):
	plt.clf()
	plt.hold(True)

	counts, bins = np.histogram(precip[:, 0], bins=30)
	plt.plot(bins[:-1], counts, color="r", label="Observed")

	counts, bins = np.histogram(sim_snowfall, bins=30)
	plt.plot(bins[:-1], counts, color="g", label="Simulated")

	plt.ylabel("Occurrences (Days)", fontsize=16)
	plt.xlabel("Total Snowfall (in)", fontsize=16)
	#plt.xlim([0.0, 2.0])
	plt.legend()
	plt.savefig(flname, DPI=300)

def plot_snowfall(flname, precip, sim_snowfall):
	plt.clf()
	plt.hold(True)
	plt.plot(sim_snowfall, color="g", alpha=0.7, label="Simulated")
	plt.plot(precip[:, 0], color="r", alpha=0.7, label="Observed")
	plt.xlabel("Time (Days)", fontsize=16)
	plt.ylabel("Total Snowfall (in)", fontsize=16)
	plt.legend()
	plt.savefig(flname, DPI=300)

def plot_weather_quality_snowfall(flname):
	plt.clf()
	snowfall = np.linspace(0.0, 3.0, 60.0)
	quality = weather_quality_snow(snowfall)
	plt.plot(snowfall, quality, color="r")
	plt.grid(True)
	plt.xlabel("Snowfall (in)", fontsize=16)
	plt.ylabel("Quality", fontsize=16)
	plt.ylim([0.0, 1.0])
	plt.legend()
	plt.savefig(flname, DPI=300)

def plot_hist_quality_snowfall(flname, precip, sim_snowfall):
	plt.clf()
	plt.hold(True)

	counts, bins = np.histogram(weather_quality_snow(precip[:, 0]), bins=10)
	counts = counts / float(sum(counts))
	plt.plot(bins[:-1], counts, color="r", label="Observed")

	counts, bins = np.histogram(weather_quality_snow(sim_snowfall), bins=10)
	counts = counts / float(sum(counts))
	plt.plot(bins[:-1], counts, color="g", label="Simulated")

	plt.ylabel("Days (Frequency)", fontsize=16)
	plt.xlabel("Quality", fontsize=16)
	plt.xlim([0.0, 1.0])
	plt.grid(True)
	plt.legend(loc="upper left")
	plt.savefig(flname, DPI=300)

def plot_scatter_temp_precip(flname, temp, precip):
	plt.clf()
	percent = []
	for snowfall, rainfall, total_precip in precip:
		if total_precip > 0.0:
			percent.append(rainfall / total_precip)
		else:
			percent.append(0.0)
	temp_percent = np.array([(temp, precip) for temp, precip in zip(temp, percent) if precip > 0.0 and precip < 1.0])
	ts = np.linspace(0.0, 60.0, 120.0)
	a = 0.4
	b = 27.0
	predicted = [1.0 / (1.0 + np.exp(-a * (T - b))) for T in ts]
	plt.scatter(temp_percent[:, 0], temp_percent[:, 1])
	plt.hold(True)
	plt.plot(ts, predicted)
	plt.xlabel("Temperature (F)", fontsize=16)
	plt.ylabel("Rainfall (% of Precip)", fontsize=16)
	plt.savefig(flname, DPI=300)

def plot_wind_speed(flname, wind_speed, sim_wind_speed=None):
	plt.clf()
	plt.hold(True)
	plt.plot(wind_speed, color="r", label="Observed")
	if sim_wind_speed != None:
		plt.plot(sim_wind_speed, color="g", label="Simulated")
	plt.xlabel("Time (Days)", fontsize=16)
	plt.ylabel("Wind Speed (mph)", fontsize=16)
	plt.legend()
	plt.savefig(flname, DPI=300)

def plot_hist_wind_speed(flname, wind_speed, sim_wind_speed=None):
	plt.clf()
	plt.hold(True)

	plt.hist(wind_speed, bins=30, color="r", alpha=0.7, label="Observed")

	if sim_wind_speed != None:
		plt.hist(sim_wind_speed, bins=30, color="g", alpha=0.7, label="Simulated")

	plt.ylabel("Occurrences (Days)", fontsize=16)
	plt.xlabel("Wind Speed (mph)", fontsize=16)
	plt.legend()
	plt.savefig(flname, DPI=300)

def plot_wind_speed_fft(flname, real, simulated=None):
	real_freq, real_amplitudes = fft(real)
	n_real_samples = len(real)
	plt.clf()
	plt.hold(True)
	plt.plot(real_freq[1:(n_real_samples+1)/2], real_amplitudes[1:(n_real_samples+1)/2], color="r", label="Real")
	
	if simulated != None:
		sim_freq, sim_amp = fft(simulated)
		n_sim_samples = len(simulated)
		plt.plot(sim_freq[1:(n_sim_samples+1)/2], sim_amp[1:(n_sim_samples+1)/2], color="k", alpha=0.7, label="Simulated")

	plt.xlabel("Frequency (cycles/day)", fontsize=16)
	plt.ylabel("Amplitude", fontsize=16)
	plt.legend(loc="upper right")
	plt.xlim([0, 0.05])
	plt.savefig(flname, DPI=300)

data_dir = sys.argv[1]
output_dir = sys.argv[2]

records = defaultdict(list)
for date in RECORDS:
	read_daily(data_dir + "/" + date + "daily.txt", records)

sbn_data = records["14848"]
sbn_data.sort()
temps = np.array([temp for (t, temp, _, _, _, _) in sbn_data])
vel = temps[1:] - temps[:len(temps) - 1]

print "Min temp: ", min(temps)
print "Avg temp: ", np.average(temps)
print "Max temp: ", max(temps)

avg = np.average(temps)
std = 1.25 * np.std(vel)
sim_temps =  simulate_temp(avg, std, 1.0, len(temps))
sim_vel = sim_temps[1:] - sim_temps[:len(sim_temps) - 1]


"""
plot_temps(output_dir + "/average_daily_temp.pdf", temps)
plot_vel(output_dir + "/average_daily_vel.pdf", vel)
plot_hist_vel(output_dir + "/average_daily_vel_hist.pdf", vel)
hist_temps(output_dir + "/average_daily_temp_hist.pdf", temps)
plot_fft(output_dir + "/average_daily_temp_fft.pdf", temps)
plot_autocorr(output_dir + "/average_daily_temp_autocorr.pdf", temps)
"""

plot_sim_temps(output_dir + "/sim_temp.pdf", temps, sim_temps)
plot_sim_vel(output_dir + "/sim_vel.pdf", vel, sim_vel)
plot_hist_sim_vel(output_dir + "/sim_vel_hist.pdf", vel, sim_vel)
plot_sim_autocorr(output_dir + "/sim_temp_autocorr.pdf", temps, sim_temps)
plot_sim_fft(output_dir + "/sim_temp_fft.pdf", temps, sim_temps)

precip = np.array([(snowfall, rainfall, total_precip) for t, _, snowfall, rainfall, total_precip, _ in sbn_data])
sim_precip, sim_snowfall, sim_rainfall = simulate_precip(2.0 * np.average(precip[:, 2]), temps, precip.shape[0])

plot_precip(output_dir + "/daily_precip.pdf", precip, sim_precip)
plot_snowfall(output_dir + "/daily_snowfall.pdf", precip, sim_snowfall)
plot_hist_precip(output_dir + "/daily_precip_hist.pdf", precip, sim_precip)
plot_hist_snowfall(output_dir + "/daily_snowfall_hist.pdf", precip, sim_snowfall)
plot_scatter_precip(output_dir + "/daily_precip_scatter.pdf", precip)
plot_scatter_temp_precip(output_dir + "/daily_temp_precip_scatter.pdf", temps, precip)
plot_autocorr(output_dir + "/daily_precip_autocorr.pdf", precip[:, 2])


wind_speeds = np.array([wind_speed for _, _, _, _, _, wind_speed in sbn_data])
k = np.sqrt(np.var(wind_speeds))
theta = np.average(wind_speeds) / k
sim_wind_speeds = simulate_wind_speed(0.0, k, theta, len(wind_speeds))

plot_wind_speed(output_dir + "/daily_wind_speeds.pdf", wind_speeds, sim_wind_speeds)
plot_hist_wind_speed(output_dir + "/daily_wind_speed_hist.pdf", wind_speeds, sim_wind_speeds)
plot_wind_speed_fft(output_dir + "/daily_wind_speed_fft.pdf", wind_speeds, sim_wind_speeds)

plot_weather_quality_snowfall(output_dir + "/weather_quality.pdf")
plot_hist_quality_snowfall(output_dir + "/daily_snowfall_quality_hist.pdf", precip, sim_snowfall)

plot_weather_quality_temp(output_dir + "/temp_quality.pdf")
plot_hist_quality_temp(output_dir + "/daily_temp_quality_hist.pdf", temps, sim_temps)

