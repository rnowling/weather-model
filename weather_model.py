import random
import numpy as np

def simulate_temp(avg, std, dt, steps, fourier_coeff):
	sim_temp = []
	noise = 0.0
	mu = 0.0
	theta = 0.5
	for i in xrange(steps):
		temp = avg
		temp += np.imag(fourier_coeff) * np.sin(-2.0 * np.pi * i / 365.0) + np.real(fourier_coeff) * np.cos(-2.0 * np.pi * i / 365.0)
		noise += theta * (mu - noise) * dt + random.gauss(0.0, std) * np.sqrt(dt)
		temp += noise
		sim_temp.append(temp)
	return np.array(sim_temp)

def snowfall_rainfall_model(temp, precip):
	a = 0.2
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
	return np.array(precips), np.array(snowfalls), np.array(rainfalls)

def simulate_wind_speed(avg, k, theta, steps, fourier_coeff):
	noise = 0.0
	sim_wind_speed = []
	for i in xrange(steps):
		wind_speed = avg
		wind_speed += np.imag(fourier_coeff) * np.sin(-2.0 * np.pi * i / 365.0) + np.real(fourier_coeff) * np.cos(-2.0 * np.pi * i / 365.0)
		noise = np.random.gamma(k, theta)
		wind_speed += noise
		wind_speed = max(0.0, wind_speed)
		sim_wind_speed.append(wind_speed)
	return np.array(sim_wind_speed)

def wind_chill_model(temp, wind_speed):
	# source: http://www.nws.noaa.gov/om/winter/windchill.shtml
	print wind_speed
	print wind_speed.min(), wind_speed.max()
	v_16 = np.power(wind_speed, 0.16)
	wc = 35.74 + 0.6215 * temp - 35.74 * v_16 + 0.4275 * temp * v_16
	return wc

def snowfall_transaction_probability(snowfall):
	a = 10.0
	b = 2.0 # inches
	c = 0.2
	quality = lambda s: 1.0 - (1.0 - c) / (1.0 + np.exp(-a * (s - b)))
	return quality(snowfall)

def rainfall_transaction_probability(rainfall):
	a = 10.0
	b = 0.5 # inches
	c = 0.4
	quality = lambda s: 1.0 - (1.0 - c) / (1.0 + np.exp(-a * (s - b)))
	return quality(rainfall)

def wind_chill_transaction_probability(wind_chill):
	a = 0.5
	b = 10.0 # F
	c = 0.2
	prob = lambda s: (1.0 - c) / (1.0 + np.exp(-a * (s - b))) + c
	return prob(wind_chill)

def wind_speed_transaction_probability(wind_speed):
	a = 0.8
	b = 15.0 # mph
	c = 0.4
	prob = lambda s: 1.0 - (1.0 - c) / (1.0 + np.exp(-a * (s - b)))
	return prob(wind_speed)

def weather_transaction_probability(wind_chill, wind_speed, snowfall, rainfall):
	a = np.minimum(snowfall_transaction_probability(snowfall), rainfall_transaction_probability(rainfall))
	b = np.minimum(wind_chill_transaction_probability(wind_chill), wind_speed_transaction_probability(wind_speed))
	return np.minimum(a, b)

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

def autocorr(series):
	shifted = series - np.average(series)
	corr = np.correlate(shifted, shifted, "full")
	corr2 = corr[:len(corr)/2] / np.var(shifted) / float(len(corr))
	return corr2

def fft(series):
	n_samples = len(series)
	freq = np.fft.fftfreq(n_samples, d=1.0)
	fft_coeff = np.fft.fft(series) / (n_samples / 2.0)
	fft_amplitudes = np.abs(fft_coeff)

	print "Median: ", np.median(fft_amplitudes) 

	for i in xrange(1, 15):
		print freq[i], 1.0 / freq[i], fft_coeff[i], fft_amplitudes[i]

	return freq, fft_amplitudes, fft_coeff