<proteins>
<aim>To generate and simulate protein binders in-silico.</aim>
<objective>max(P(functional))</objective> # in the wet lab
<input>A target protein.</input>
<output>A _specific_ and _selective protein binder.</output>
<language>Python</language>>
<infrastructure>Modal</infrastructure> # budget $30 USD
<tools:{open-source:True}>
  # Specificity
  - **RFDiffusion**: Backbone structure generation.
  - **ProteinMPNN**: Backbone structure to sequence conversion.
  - **Boltz-2**: Binder-target affinity.
  # Selectivity
  - **FoldSeek**: Flagging similar structures in the proteome to the target.
  - **Chai-1**: Binder-proteome affinity (versus similar structures).
</tools>
</proteins>
