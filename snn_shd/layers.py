import torch
from snntorch import Synaptic


class CramerSynaptic(Synaptic):

    def _base_state_function(self, input_):
        base_fn_mem = (
            self.beta.clamp(0, 1) * self.mem + (1 - self.beta.clamp(0, 1)) * self.syn
        )
        base_fn_syn = self.alpha.clamp(0, 1) * self.syn + input_
        return base_fn_syn, base_fn_mem

    def _base_state_reset_zero(self, input_):
        base_fn_mem = (
            self.beta.clamp(0, 1) * self.mem + (1 - self.beta.clamp(0, 1)) * self.syn
        )
        base_fn_syn = self.alpha.clamp(0, 1) * self.syn + input_
        return 0, base_fn_mem

    def _base_zero(self, input_):
        syn, mem = self._base_state_function(input_)
        syn2, mem2 = self._base_state_reset_zero(input_)
        syn -= syn2 * self.reset
        mem -= mem2 * self.reset
        mem += (1 - self.beta.clamp(0, 1)) * self.syn * self.reset
        return syn, mem

    def forward(self, input_, syn=None, mem=None):

        if not syn == None:
            self.syn = syn

        if not mem == None:
            self.mem = mem

        if self.init_hidden and (not mem == None or not syn == None):
            raise TypeError(
                "`mem` or `syn` should not be passed as an argument while `init_hidden=True`"
            )

        if not self.syn.shape == input_.shape:
            self.syn = torch.zeros_like(input_, device=self.syn.device)

        if not self.mem.shape == input_.shape:
            self.mem = torch.zeros_like(input_, device=self.mem.device)

        self.reset = self.mem_reset(self.mem)

        if self.inhibition:
            spk = self.fire_inhibition(self.mem.size(0), self.mem)  # batch_size
        else:
            spk = self.fire(self.mem)

        self.syn, self.mem = self.state_function(input_)

        if self.state_quant:
            self.mem = self.state_quant(self.mem)
            self.syn = self.state_quant(self.syn)

        if not self.reset_delay:
            # reset membrane potential _right_ after spike
            do_reset = (
                spk / self.graded_spikes_factor - self.reset
            )  # avoid double reset
            if self.reset_mechanism_val == 0:  # reset by subtraction
                self.mem = self.mem - do_reset * self.threshold
            elif self.reset_mechanism_val == 1:  # reset to zero
                self.mem = self.mem - do_reset * self.mem

        if self.output:
            return spk, self.syn, self.mem
        elif self.init_hidden:
            return spk
        else:
            return spk, self.syn, self.mem
