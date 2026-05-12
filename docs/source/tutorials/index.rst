Tutorials
=========

Interactive notebooks demonstrating MaldiDeepKit workflows end to end.
Each notebook is self-contained and can be downloaded from the
`GitHub repository <https://github.com/EttoreRocchi/MaldiDeepKit/tree/main/notebooks>`_.

.. toctree::
   :maxdepth: 2

   notebooks/01_quick_start
   notebooks/02_model_comparison
   notebooks/03_attention_interpretation
   notebooks/04_full_pipeline
   notebooks/05_uncertainty

Example Workflows
-----------------

Single-Architecture Baseline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from sklearn.model_selection import StratifiedKFold, cross_val_score
   from sklearn.pipeline import Pipeline
   from sklearn.preprocessing import StandardScaler
   from maldideepkit import MaldiMLPClassifier

   rng = np.random.default_rng(0)
   X = rng.standard_normal((400, 6000)).astype("float32")
   y = rng.integers(0, 2, size=400)

   pipe = Pipeline([
       ("scaler", StandardScaler()),
       ("clf", MaldiMLPClassifier(random_state=0)),
   ])

   cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
   scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
   print(f"CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

Cross-Architecture Comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from maldideepkit import (
       MaldiCNNClassifier,
       MaldiMLPClassifier,
       MaldiResNetClassifier,
       MaldiTransformerClassifier,
   )

   classifiers = {
       "mlp":         MaldiMLPClassifier(random_state=0),
       "cnn":         MaldiCNNClassifier(random_state=0),
       "resnet":      MaldiResNetClassifier(random_state=0),
       "transformer": MaldiTransformerClassifier(random_state=0),
   }

   for name, clf in classifiers.items():
       clf.fit(X, y)
       print(f"{name}: {clf.score(X, y):.3f}")

Full MaldiSuite Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from maldiamrkit import MaldiSet
   from maldibatchkit import SpeciesAwareComBat
   from maldibatchkit.integrations import MaldiSetAdapter
   from maldideepkit import MaldiCNNClassifier

   ds = MaldiSet.from_directory(
       "spectra/", "metadata.csv",
       aggregate_by={"antibiotics": "Ciprofloxacin"},
       n_jobs=-1,
   )

   # Optional batch-effect correction across acquisition sites
   adapter = MaldiSetAdapter(batch_column="Batch", species_column="Species")
   ds = adapter.correct(ds, SpeciesAwareComBat)

   clf = MaldiCNNClassifier(random_state=0).fit(ds, ds.y.squeeze())
   clf.save("ciprofloxacin_model")
   preds = clf.predict(ds)
