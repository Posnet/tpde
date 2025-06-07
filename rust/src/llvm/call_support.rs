//! Support for advanced calling convention features.
//!
//! This module provides functionality for handling byval, sret, varargs,
//! and other advanced calling convention attributes.

use inkwell::attributes::{Attribute, AttributeLoc};
use inkwell::values::CallSiteValue;
use crate::x64::calling_convention::ArgAttribute;

/// Check if a function call is a varargs call.
pub fn is_varargs_call(call_site: CallSiteValue) -> bool {
    if let Some(func) = call_site.get_called_fn_value() {
        func.get_type().is_var_arg()
    } else {
        false
    }
}

/// Get the parameter attributes for a call argument.
pub fn get_param_attributes(call_site: CallSiteValue, param_idx: u32) -> ArgAttribute {
    // Check for byval attribute
    let byval_kind_id = Attribute::get_named_enum_kind_id("byval");
    if let Some(_byval_attr) = call_site.get_enum_attribute(AttributeLoc::Param(param_idx), byval_kind_id) {
        // For byval, we need to get the type info from the attribute
        // In LLVM, byval carries type information
        // For now, we'll need to handle this specially
        log::debug!("Found byval attribute on param {}", param_idx);
        
        // Try to get size and alignment from other attributes
        let align_kind_id = Attribute::get_named_enum_kind_id("align");
        let align = if let Some(align_attr) = call_site.get_enum_attribute(AttributeLoc::Param(param_idx), align_kind_id) {
            align_attr.get_enum_value() as u32
        } else {
            8 // Default alignment
        };
        
        // For byval, size needs to be determined from the pointed-to type
        // This is a limitation of inkwell - we'll need to handle this specially
        return ArgAttribute::ByVal { size: 0, align }; // Size will be filled in by caller
    }
    
    // Check for sret attribute
    let sret_kind_id = Attribute::get_named_enum_kind_id("sret");
    if call_site.get_enum_attribute(AttributeLoc::Param(param_idx), sret_kind_id).is_some() {
        log::debug!("Found sret attribute on param {}", param_idx);
        return ArgAttribute::StructRet;
    }
    
    // Check for zext attribute
    let zext_kind_id = Attribute::get_named_enum_kind_id("zeroext");
    if call_site.get_enum_attribute(AttributeLoc::Param(param_idx), zext_kind_id).is_some() {
        log::debug!("Found zeroext attribute on param {}", param_idx);
        return ArgAttribute::ZeroExt;
    }
    
    // Check for sext attribute
    let sext_kind_id = Attribute::get_named_enum_kind_id("signext");
    if call_site.get_enum_attribute(AttributeLoc::Param(param_idx), sext_kind_id).is_some() {
        log::debug!("Found signext attribute on param {}", param_idx);
        return ArgAttribute::SignExt;
    }
    
    ArgAttribute::None
}

/// Get the byval type size for a parameter.
/// This is a workaround for inkwell not exposing getParamByValType.
pub fn get_byval_size(_call_site: CallSiteValue, _param_idx: u32, arg_value: inkwell::values::BasicValueEnum) -> u32 {
    // For byval arguments, we need the size of the pointed-to type
    if let inkwell::values::BasicValueEnum::PointerValue(_ptr) = arg_value {
        // Get the element type that the pointer points to
        // In LLVM 15+, pointers are opaque, so we need context-specific info
        // For now, we'll use a conservative estimate
        log::warn!("byval size estimation not fully implemented - using conservative size");
        64 // Conservative estimate
    } else {
        8 // Default
    }
}

/// Handle special calling convention setup for varargs functions.
pub fn setup_varargs_call(cc_assigner: &mut crate::x64::calling_convention::SysVAssigner, varargs_start: u32) {
    // For System V ABI, we need to set up AL register with the number of
    // XMM registers used for varargs
    // This is handled later in the actual call emission
    log::debug!("Setting up varargs call starting at arg {}", varargs_start);
    cc_assigner.set_must_assign_stack();
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_attribute_kind_ids() {
        // Verify that we can get the correct kind IDs for attributes
        let byval_id = Attribute::get_named_enum_kind_id("byval");
        let sret_id = Attribute::get_named_enum_kind_id("sret");
        let zext_id = Attribute::get_named_enum_kind_id("zeroext");
        let sext_id = Attribute::get_named_enum_kind_id("signext");
        
        // These should all be non-zero (valid attributes)
        assert_ne!(byval_id, 0);
        assert_ne!(sret_id, 0);
        assert_ne!(zext_id, 0);
        assert_ne!(sext_id, 0);
        
        // They should all be different
        assert_ne!(byval_id, sret_id);
        assert_ne!(byval_id, zext_id);
        assert_ne!(byval_id, sext_id);
    }
}