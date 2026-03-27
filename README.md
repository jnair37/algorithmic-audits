# ExplanAudit: Algorithmic Auditing with Explanations

Welcome to ExplanAudit! This is a senior thesis project by Jade Nair '26 for the Computer Science Department at Harvard College.

## About

As machine learning algorithms are used in increasingly consequential settings, the idea of auditing algorithms, or evaluating them in a structured manner against specific fairness or safety criteria, is gradually gaining traction. With a lack of federal regulation in the United States and industry standards currently emerging, advocates for more widespread, comprehensive auditing must navigate a complex technical and policy ecosystem. This thesis investigates the current landscape of algorithmic auditing in the United States, examines the potential for applications of interpretable machine learning to create more comprehensive audits, and develops ExplanAudit (this repository!), a software tool to incorporate interpretability while supporting scalability, accessibility, and visibility for this work. This is accomplished through a user study of n=15 algorithmic auditors based in the United States with both qualitative and demonstration components. We find that, in contrast with the general population, the examined population of auditors does not become overconfident as a result of exposure to the explanations; as a result, our toolkit supports a variety of interpretable methods for auditing and is found at https://gradio-app-805328664778.us-central1.run.app/. Future work could expand this study to a larger sample of auditors, extend the software to train new auditors, or apply the tool to audit specific behaviors of open-source models.

## Instructions for use

The tool is deployed to the web through Google Cloud Platform here: https://gradio-app-805328664778.us-central1.run.app/
However, there is currently significant latency when using this site, so in case of difficulties we provide an alternative way to run through Google Colab. 

If you open **ExplanAudit.ipynb** in Colab and run the notebook step by step, you should be able to run the interface on your browser at a faster processing speed. 
(Note that you will need a HuggingFace token and if you want to use Llama, you will need to be approved to use that model as well.)