#/bin/bash


get_gene_species () {
    bn=$(basename $1)
    gene_species=$(awk -F'_' '{printf "%s_%s",$1,$2}' <(echo "$bn"))
    echo $gene_species
}
