/*********************************************************************************
 *
 * Inviwo - Interactive Visualization Workshop
 *
 * Copyright (c) 2018 Inviwo Foundation
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *********************************************************************************/

#include <modules/test_module/test_modulemodule.h>
#include <test_module/processor.h>
#include <modules/test_module/my_test_class_w_skeleton.h>

namespace inviwo {

test_moduleModule::test_moduleModule(InviwoApplication* app) : InviwoModule(app, "test_module") {
    registerProcessor<processor>();
    registerProcessor<my_test_class_w_skeleton>();
    // Add a directory to the search path of the Shadermanager
    // ShaderManager::getPtr()->addShaderSearchPath(getPath(ModulePath::GLSL));

    // Register objects that can be shared with the rest of inviwo here:

    // Processors
    // registerProcessor<test_moduleProcessor>();

    // Properties
    // registerProperty<test_moduleProperty>();

    // Readers and writes
    // registerDataReader(util::make_unique<test_moduleReader>());
    // registerDataWriter(util::make_unique<test_moduleWriter>());

    // Data converters
    // registerRepresentationConverter(util::make_unique<test_moduleDisk2RAMConverter>());

    // Ports
    // registerPort<test_moduleOutport>();
    // registerPort<test_moduleInport>();

    // PropertyWidgets
    // registerPropertyWidget<test_modulePropertyWidget, test_moduleProperty>("Default");

    // Dialogs
    // registerDialog<test_moduleDialog>(test_moduleOutport);

    // Other things
    // registerCapabilities(util::make_unique<test_moduleCapabilities>());
    // registerSettings(util::make_unique<test_moduleSettings>());
    // registerMetaData(util::make_unique<test_moduleMetaData>());
    // registerPortInspector("test_moduleOutport", "path/workspace.inv");
    // registerProcessorWidget(std::string processorClassName, std::unique_ptr<ProcessorWidget> processorWidget); 
    // registerDrawer(util::make_unique_ptr<test_moduleDrawer>());
}

}  // namespace inviwo
