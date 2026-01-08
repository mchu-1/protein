<proteins>
<aim>To generate and simulate protein binders in-silico.</aim>
<input>A target protein.</input>
<output>A _specific_ and _selective protein binder.</output>
<language>Python</language>>
<infrastructure>Modal</infrastructure>
<tools:{open-source:True}>
  # Specificity
  - **RFDiffusion**: Backbone structure generation.
  - **ProteinMPNN**: Backbone structure to sequence conversion.
  # Selectivity
  - **FoldSeek**: Flagging similar structures in the proteome to the target.
  # Affinity Determination
  - **Boltz-2**
</tools>
</proteins>
