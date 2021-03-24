

def flatten_dict(dict_):
	"""Remove nested hierarchies in dict ignoring keys in higher levels and flatten the dict"""
	
	def _flatten(dict_, key_str=''): 
	    if type(dict_) == dict: 
	        for k in dict_: 
	            yield from _flatten(dict_[k], str(k)) 
	    else: 
	        yield key_str, dict_

	return dict(_flatten(dict_))
