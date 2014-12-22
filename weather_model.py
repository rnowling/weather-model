import random
import numpy as np

def simulate_temp(avg, std, dt, steps):
	sim_temp = []
	noise = 0.0
	mu = 0.0
	theta = 0.3
	for i in xrange(steps):
		temp = avg
		temp += 22.624 * np.sin(-2.0 * np.pi * i / 365.0) + 7.829 * np.cos(-2.0 * np.pi * i / 365.0)
		noise += theta * (mu - noise) * dt + random.gauss(0.0, std) * np.sqrt(dt)
		temp += noise
		sim_temp.append(temp)
	return np.array(sim_temp)

def snowfall_rainfall_model(temp, precip):
	a = 0.4
	b = 27.0
	percent = lambda t: 1.0 / (1.0 + np.exp(-a * (t - b)))
	snowfall = 10 * (1.0 - percent(temp)) * precip
	rainfall = percent(temp) * precip
	return snowfall, rainfall

def simulate_precip(avg, temps, days):
	precips = []
	snowfalls = []
	rainfalls = []
	for i in xrange(days):
		precip = random.expovariate(1.0 / avg)
		snowfall, rainfall = snowfall_rainfall_model(temps[i], precip)
		precips.append(precip)
		snowfalls.append(snowfall)
		rainfalls.append(rainfall)
	return np.array(precips), np.array(snowfalls), np.array(rainfall)

def simulate_wind_speed(avg, k, theta, steps):
	noise = 0.0
	sim_wind_speed = []
	for i in xrange(steps):
		wind_speed = avg
		wind_speed += -1.3142 * np.sin(-2.0 * np.pi * i / 365.0) - 1.5238 * np.cos(-2.0 * np.pi * i / 365.0)
		noise = np.random.gamma(k, theta)
		wind_speed += noise
		sim_wind_speed.append(wind_speed)
	return np.array(sim_wind_speed)

def wind_chill_model(temp, wind_speed):
	v_16 = np.pow(wind_speed, 0.16)
	wc = 35.74 + 0.6215 * temp - 35.74 * v_16 + 0.4275 * temp * v_16
	return wc

def weather_quality_snow(snowfall):
	a = 10.0
	b = 0.75 # inches
	c = 0.2
	quality = lambda s: 1.0 - (1.0 - c) / (1.0 + np.exp(-a * (s - b)))
	return quality(snowfall)

def weather_quality_temp(temp):
	a = 0.5
	b = 25.0 # F
	c = 0.4
	quality = lambda s: (1.0 - c) / (1.0 + np.exp(-a * (s - b))) + c
	return quality(temp)

def weather_quality_wind_chill(wind_chill):
	a = 0.5
	b = 10.0 # F
	c = 0.4
	quality = lambda s: (1.0 - c) / (1.0 + np.exp(-a * (s - b))) + c
	return quality(temp)

def read_daily(flname, records):
	fl = open(flname)
	next(fl)
	for ln in fl:
		cols = ln.strip().split(",")
		WBAN = cols[0]
		date = int(cols[1])
		#temp_max = cols[2]
		#temp_min = cols[4]
		temp_avg = cols[6]

		try:
			snowfall = float(cols[28])
		except:
			snowfall = 0.0

		try:
			rainfall = float(cols[30])
		except:
			rainfall = 0.0

		total_precip = 0.1 * snowfall + rainfall
		
		try:
			avg_wind_speed = float(cols[40])
		except:
			avg_wind_speed = 0.0

		try:
			temp_avg = float(temp_avg)
		except:
			continue

		records[WBAN].append((date, temp_avg, snowfall, rainfall, total_precip, avg_wind_speed))

	fl.close()

def fft(series):
	n_samples = len(series)
	freq = np.fft.fftfreq(n_samples, d=1.0)
	fft_coeff = np.fft.fft(series) / (n_samples / 2.0)
	fft_amplitudes = np.abs(fft_coeff)

	print "Median: ", np.median(fft_amplitudes) 

	for i in xrange(1, 15):
		print freq[i], 1.0 / freq[i], fft_coeff[i], fft_amplitudes[i]

	return freq, fft_amplitudes