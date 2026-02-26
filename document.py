from pydantic import BaseModel, Field, field_validator


class BillOfLading(BaseModel):
    shipper_name: str = Field(description="Name of the shipper")
    consignee_name: str = Field(description="Name of the consignee")
    vessel: str = Field(description="Vessel name and voyage number")
    port_of_loading: str = Field(description="Port where the goods are loaded")
    port_of_discharge: str = Field(description="Port where the goods are discharged")
    container_numbers: list[str] = Field(description="List of all container numbers")
    total_gross_weight_kg: float = Field(description="Total gross weight in kilograms")

    @field_validator("container_numbers")
    @classmethod
    def deduplicate_containers(cls, v):
        return list(dict.fromkeys(v))
