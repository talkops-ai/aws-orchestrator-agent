variable "name" { type = string }
variable "create_vpc" { type = bool default = true }
variable "cidr" { type = string default = "10.0.0.0/16" }
variable "azs" { type = list(string) default = ["ap-south-1a", "ap-south-1b", "ap-south-1c"] }
variable "public_subnets" { type = list(string) default = [] }
variable "private_app_subnets" { type = list(string) default = [] }
variable "private_data_subnets" { type = list(string) default = [] }
variable "enable_nat_gateway" { type = bool default = true }
variable "single_nat_gateway" { type = bool default = false }