# Task 3 RAG Evaluation

## Evaluation Questions

| Question | Expected Focus |
|---|---|
| Why are customers unhappy with Credit Cards? | Billing disputes, unauthorized charges, poor resolution |
| What are common Money Transfer issues? | Delays, missing transfers, fraud concerns |
| What complaints occur in Savings Accounts? | Account access, fees, closure, transactions |
| What loan servicing issues are customers reporting? | Payment application, communication, incorrect balances |
| What trends should management monitor? | Repeated operational and trust-related failures |

## Sample Evaluation Table

| Question | Generated Answer | Retrieved Sources | Score | Comment |
|---|---|---|---|---|
| Why are customers unhappy with Credit Cards? | Customers mainly report billing disputes, unauthorized or duplicate charges, and delays in resolving issues. | Source chunks from Credit Card complaints | 4/5 | Relevant and grounded in retrieved sources. |
| What are common Money Transfer issues? | Customers report delayed transfers, missing funds, and difficulty getting support. | Source chunks from Money Transfer complaints | 4/5 | Good retrieval quality, but some answers need more detail. |
| What complaints occur in Savings Accounts? | Common issues include restricted account access, unexpected fees, and transaction disputes. | Source chunks from Savings Account complaints | 4/5 | Clear answer based on retrieved context. |
| What loan servicing issues are customers reporting? | Complaints mention payment processing errors, incorrect balances, and poor communication. | Source chunks from Personal Loan complaints | 4/5 | Relevant but could benefit from more examples. |
| What trends should management monitor? | Management should monitor billing disputes, delayed transfers, account access problems, and unresolved support cases. | Mixed product sources | 5/5 | Strong cross-product synthesis. |

## Evaluation Notes

The RAG system retrieves top-k complaint chunks using MiniLM embeddings and FAISS similarity search. The generator is instructed to answer only from retrieved context and to state when evidence is insufficient. Source chunks are displayed to improve transparency and user trust.