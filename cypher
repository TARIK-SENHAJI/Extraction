LOAD CSV WITH HEADERS FROM 'file:///Clean.csv' AS row 
FIELDTERMINATOR ';'
WITH row WHERE row.h IS NOT NULL AND row.t IS NOT NULL 
MERGE (n:Source {name: row.h}) 
MERGE (m:Destination {name: row.t}) 
MERGE (n)-[:TO {type: row.r}]->(m)
