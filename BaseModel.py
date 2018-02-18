
class BaseModel(object):
	def __init__(self):
		pass

	def false_positive_rate(self, actual, predicted):
		assert len(actual) == len(predicted)
		
		fp_count = 0
		p_count = 0
		for i in range(len(actual)):
			if actual[i] == 0 and predicted[i] == 1:
				fp_count += 1
			if (predicted[i] == 1):
				p_count += 1

		return float(fp_count)/float(p_count)

	def false_negative_rate(self, actual, predicted):
		assert len(actual) == len(predicted)
		
		fn_count = 0
		n_count = 0
		for i in range(len(actual)):
			if actual[i] == 1 and predicted[i] == 0:
				fn_count += 1
			if (predicted[i] == 0):
				n_count += 1

		return float(fn_count)/float(n_count)

	def misclassification_rate(self, actual, predicted):
		assert len(actual) == len(predicted)
		
		count = 0
		for i in range(len(actual)):
			if (actual[i] == 1 and predicted[i] == 0) or (actual[i] == 0 and predicted[i] == 1):
				count += 1

		return float(count)/len(actual)