# Normalization constants for translation components (Y and Z)
# Replace these with actual values from your dataset
MEAN_Y = 0.0
STD_Y = 1.0
MEAN_Z = 0.0
STD_Z = 1.0


def canonicalize_translation_norm_stats(stats=None):
	"""Return translation normalization stats using the mean/std key schema.

	Older preprocessing caches and configs used trans_mean_y/trans_std_y keys,
	while the training and evaluation code now reads mean_y/std_y directly.
	This helper accepts either schema and always returns the canonical one.
	"""
	stats = stats or {}

	return {
		"mean_y": stats.get("mean_y", stats.get("trans_mean_y", MEAN_Y)),
		"std_y": stats.get("std_y", stats.get("trans_std_y", STD_Y)),
		"mean_z": stats.get("mean_z", stats.get("trans_mean_z", MEAN_Z)),
		"std_z": stats.get("std_z", stats.get("trans_std_z", STD_Z)),
	}
