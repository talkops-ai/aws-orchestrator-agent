data "aws_region" "current" {}
locals {
  name_prefix = "$${var.name}-%s-Production-$${data.aws_region.current.id}"
  common_tags = merge({ Environment = "Production", ManagedBy = "Terraform", Compliance = "HIPAA", CostCenter = "Infrastructure" }, var.tags)
}
resource "aws_vpc" "this" {
  count = var.create_vpc ? 1 : 0
  cidr_block = var.cidr
  enable_dns_hostnames = true
  enable_dns_support = true
  tags = merge(local.common_tags, { "Name" = format(local.name_prefix, "vpc") }, var.vpc_tags)
}