library(dplyr)

# Read and drop the first column
data12 <- read.csv("diabetes_012_health_indicators_BRFSS2015.csv") %>% select(-1) %>% mutate(Source = "data12")
dataBinary <- read.csv("diabetes_binary_5050split_health_indicators_BRFSS2015.csv") %>% select(-1) %>% mutate(Source = "dataBinary")
data50 <- read.csv("diabetes_binary_health_indicators_BRFSS2015.csv") %>% select(-1) %>% mutate(Source = "data50")

# Combine all data
all_data <- bind_rows(data12, dataBinary, data50)

# Find rows that appear in only one dataset
duplicates <- all_data %>%
  group_by(across(-Source)) %>%
  summarise(n = n_distinct(Source), .groups = "drop") %>%
  filter(n == 1)  # Unique to only one source

# Extract those unique rows
unique_rows <- semi_join(all_data, duplicates, by = colnames(duplicates)[1:(ncol(duplicates) - 1)])

# Save to CSV
write.csv(unique_rows, "unique_rows_across_datasets.csv", row.names = FALSE)

