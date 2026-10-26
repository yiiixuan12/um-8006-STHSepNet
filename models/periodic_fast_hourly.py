"""Same fast56-token interface, using existing causal hourly indexing."""
from models.periodic_hourly import HourlyPeriodicBertV4
from models.periodic_bert_fast import FastPeriodicBert


class HourlyFastPeriodicBert(HourlyPeriodicBertV4,FastPeriodicBert):
    """Hourly forward/token/time indexing + fast chunk/shortlist implementation.

    MRO preserves HourlyPeriodicBertV4.history_length=240 and its candidate
    operator. Constructor tensors/order/RNG stay unchanged. No new weights.
    """
    @property
    def metadata(self):
        return {**super().metadata,'interface_version':'periodic_fast_hourly_v1'}
